"""Verified-date Brunei prayer schedule lookup (free, read-only).

Mirrors KHEU's annual official taqwim as structured monthly records; the host
is a community-maintained mirror, NOT an independently verified live KHEU API.
Every answer names the precise date, district and mirror provenance.
No guesses, no paid APIs, no user-controlled URLs, no persistent user data.
"""
import asyncio
import datetime as dt
import json
import re
from zoneinfo import ZoneInfo

import httpx

TIMEZONE = ZoneInfo("Asia/Brunei")
MIRROR_BASE = "https://simplesolat-data.netlify.app/prayer-times/BN"
GITHUB_BASE = (
    "https://raw.githubusercontent.com/ragibkl/"
    "simplesolat-data/main/data/prayer-times/BN"
)
ZONES = {
    "BRN01": "Brunei-Muara",
    "BRN02": "Tutong",
    "BRN03": "Belait",
    "BRN04": "Temburong",
}
PRAYERS = {
    "subuh": ("fajr", "Subuh"),
    "fajr": ("fajr", "Subuh"),
    "fajar": ("fajr", "Subuh"),
    "imsak": ("imsak", "Imsak"),
    "syuruk": ("syuruk", "Syuruk"),
    "dhuha": ("dhuha", "Dhuha"),
    "zohor": ("dhuhr", "Zohor"),
    "zuhur": ("dhuhr", "Zohor"),
    "dhuhr": ("dhuhr", "Zohor"),
    "asar": ("asr", "Asar"),
    "asr": ("asr", "Asar"),
    "maghrib": ("maghrib", "Maghrib"),
    "isyak": ("isha", "Isyak"),
    "isya": ("isha", "Isyak"),
    "isha": ("isha", "Isyak"),
}
MONTHS = {
    "januari": 1, "january": 1, "februari": 2, "february": 2,
    "mac": 3, "march": 3, "april": 4, "mei": 5, "may": 5,
    "jun": 6, "june": 6, "julai": 7, "july": 7, "ogos": 8,
    "august": 8, "september": 9, "oktober": 10, "october": 10,
    "november": 11, "nov": 11, "disember": 12, "december": 12, "dec": 12,
}
MAX_RESPONSE_BYTES = 100_000
REQUIRED_KEYS = ("imsak", "fajr", "syuruk", "dhuhr", "asr", "maghrib", "isha")


def recognize_prayer_question(question: str) -> bool:
    """Only route actual Brunei prayer-time questions through this special path."""
    if not isinstance(question, str) or len(question) > 800:
        return False
    q = question.casefold()
    location = re.search(
        r"\b(?:brunei|brunei-muara|muara|tutong|belait|temburong|"
        r"bandar seri begawan|seria|kuala belait)\b", q
    )
    timing = re.search(
        r"\b(?:waktu|masa|jam|pukul|time|jadual|schedule|bila|"
        r"prayer|solat|sembahyang|azan|adzan)\b", q
    )
    named_prayer = re.search(r"\b(?:" + "|".join(PRAYERS) + r")\b", q)
    return bool(location and (named_prayer or timing and re.search(
        r"\b(?:solat|sembahyang|prayer|azan|adzan)\b", q)))


def prayer_requested(question: str) -> tuple[str, str]:
    q = question.casefold()
    for word in re.findall(r"[a-z]+", q):
        if word in PRAYERS:
            return PRAYERS[word]
    return "all", "Semua waktu solat"


def requested_zones(question: str) -> list[str]:
    q = question.casefold()
    result = []
    if re.search(r"\b(?:brunei[\s-]muara|bandar seri begawan|muara|bsb)\b", q):
        result.append("BRN01")
    if re.search(r"\btutong\b", q):
        result.append("BRN02")
    if re.search(r"\b(?:belait|seria|kuala belait)\b", q):
        result.append("BRN03")
    if re.search(r"\btemburong\b", q):
        result.append("BRN04")
    # Bare "Brunei" has four different official district schedules.
    return result or list(ZONES)


def requested_date(question: str, *, now: dt.datetime | None = None) -> dt.date | None:
    """Local Brunei clock; never treat a missing/invalid explicit date as today."""
    current = (now.astimezone(TIMEZONE) if now is not None
               else dt.datetime.now(TIMEZONE))
    q = question.casefold()
    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if iso:
        try:
            return dt.date(*(int(x) for x in iso.groups()))
        except ValueError:
            return None
    slashed = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})\b", q)
    if slashed:
        try:
            day, month, year = (int(x) for x in slashed.groups())
            return dt.date(year, month, day)
        except ValueError:
            return None
    month_names = "|".join(MONTHS)
    written = re.search(
        r"\b(\d{1,2})\s+(" + month_names + r")(?:\s+(20\d{2}))?\b", q
    )
    if written:
        try:
            return dt.date(
                int(written.group(3) or current.year),
                MONTHS[written.group(2)], int(written.group(1)))
        except ValueError:
            return None
    if re.search(r"\b(?:esok|tomorrow)\b", q):
        return current.date() + dt.timedelta(days=1)
    if re.search(r"\b(?:semalam|yesterday)\b", q):
        return current.date() - dt.timedelta(days=1)
    return current.date()


def _valid_schedule(rows, target: dt.date) -> dict | None:
    """Validate date + every prayer HH:MM; never silently use a neighbouring day."""
    if not isinstance(rows, list) or len(rows) > 35:
        return None
    found = [x for x in rows if isinstance(x, dict)
             and x.get("date") == target.isoformat()]
    if len(found) != 1:
        return None
    item = found[0]
    for k in REQUIRED_KEYS:
        v = item.get(k)
        if (not isinstance(v, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", v)):
            return None
    return item


def _source_urls(zone: str, target: dt.date) -> tuple[str, str]:
    # All dynamic path parts derive exclusively from validated local values.
    if zone not in ZONES or not 2020 <= target.year <= 2035:
        raise ValueError("Zone/year unsupported")
    suffix = f"{zone}/{target.year:04d}-{target.month:02d}.json"
    return f"{MIRROR_BASE}/{suffix}", f"{GITHUB_BASE}/{suffix}"


async def lookup_brunei_prayer_time(
    question: str, *, client: httpx.AsyncClient | None = None,
    now: dt.datetime | None = None,
) -> str:
    target = requested_date(question, now=now)
    if target is None:
        return "Tarikh yang diberikan tidak sah. Contoh: /deepdive waktu Subuh Brunei 18 November 2026."
    if target.year < 2020 or target.year > 2035:
        return "Jadual untuk tahun tersebut tidak tersedia dalam sumber yang dibenarkan."
    zones = requested_zones(question)
    prayer_key, prayer_name = prayer_requested(question)

    async def fetch_zone(session, zone):
        primary, backup = _source_urls(zone, target)
        for url in (primary, backup):
            try:
                response = await session.get(
                    url, timeout=httpx.Timeout(6, connect=3),
                    follow_redirects=False,
                )
                if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                    continue
                if "json" not in response.headers.get("content-type", "").lower():
                    continue
                schedule = _valid_schedule(response.json(), target)
                if schedule is not None:
                    return zone, schedule, url
            except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return zone, None, None

    async def fetch_all(session):
        return await asyncio.gather(*(fetch_zone(session, zone) for zone in zones))

    if client is not None:
        results = await fetch_all(client)
    else:
        async with httpx.AsyncClient() as session:
            results = await fetch_all(session)

    names = {
        "imsak": "Imsak", "fajr": "Subuh", "syuruk": "Syuruk",
        "dhuha": "Dhuha", "dhuhr": "Zohor", "asr": "Asar",
        "maghrib": "Maghrib", "isha": "Isyak",
    }
    lines = [f"Waktu {prayer_name} — Brunei, {target.strftime('%d/%m/%Y')} (GMT+8):"]
    sources = []
    for zone, schedule, source in results:
        if schedule is None:
            lines.append(f"• {ZONES[zone]}: tidak dapat disahkan daripada jadual tersedia.")
            continue
        if prayer_key == "all":
            fields = ("imsak", "fajr", "syuruk", "dhuhr", "asr", "maghrib", "isha")
            info = ", ".join(
                f"{names[k]} {schedule[k]}" for k in fields
            )
        else:
            info = f"{schedule[prayer_key]} pagi" if prayer_key in ("fajr", "imsak", "syuruk", "dhuha") else schedule[prayer_key]
        lines.append(f"• {ZONES[zone]}: {info}")
        sources.append(source)
    lines.append(
        "\nSumber data: cermin komuniti simplesolat-data daripada takwim KHEU "
        "(bukan capaian langsung sistem rasmi KHEU). "
        "Untuk keputusan ibadah penting, rujuk jadual rasmi KHEU."
    )
    if sources:
        lines.append("Data: " + sources[0])
    return "\n".join(lines)
