"""Shared tools, actors, and logging for the 05_smart_building distributed demo."""

import random
from datetime import datetime

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    Actor,
    DelegationRequest,
    DelegationResult,
    Hub,
    LoopDetector,
    SchedulerTriggerFired,
    Signal,
    TokenMonitor,
)
from autogen.beta.tools.final import tool

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

PORTS = {"climate": 8911, "operations": 8912}

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"

# ---------------------------------------------------------------------------
# HVAC tools
# ---------------------------------------------------------------------------


@tool
async def read_temperature(zone: str) -> str:
    """Read current temperature for a building zone.

    Args:
        zone: Building zone (lobby, office-floor-2, office-floor-3, server-room, parking).
    """
    base_temps = {"lobby": 71.0, "office-floor-2": 72.0, "office-floor-3": 70.5, "server-room": 78.0, "parking": 68.0}
    base = base_temps.get(zone, 71.0)
    temp = base + random.choice([-8, -6, 10, 12, 15]) if random.random() < 0.3 else base + random.uniform(-2.0, 2.0)
    status = "NORMAL" if 68.0 <= temp <= 74.0 else "OUT OF RANGE"
    return f"Zone: {zone}\n  Temperature: {temp:.1f}\u00b0F\n  Status: {status}"


@tool
async def adjust_thermostat(zone: str, target_temp: float) -> str:
    """Adjust thermostat for a building zone.

    Args:
        zone: Building zone to adjust.
        target_temp: Target temperature in Fahrenheit.
    """
    return f"Thermostat adjusted for {zone}.\n  Target: {target_temp:.0f}\u00b0F\n  ETA: {random.randint(3, 12)} min"


@tool
async def check_air_quality(zone: str) -> str:
    """Check air quality metrics for a building zone.

    Args:
        zone: Building zone to check.
    """
    aqi = random.randint(15, 65)
    co2 = random.randint(380, 800)
    humidity = random.uniform(35.0, 60.0)
    return f"Air Quality — {zone}\n  AQI: {aqi}\n  CO2: {co2} ppm\n  Humidity: {humidity:.1f}%"


@tool
async def set_ventilation_mode(zone: str, mode: str) -> str:
    """Set ventilation mode for a building zone.

    Args:
        zone: Building zone to configure.
        mode: Ventilation mode (normal, boost, economy).
    """
    fan = {"normal": "60%", "boost": "100%", "economy": "30%"}
    return f"Ventilation in {zone}: {mode.upper()}, fan speed: {fan.get(mode, '60%')}"


hvac_tools = [read_temperature, adjust_thermostat, check_air_quality, set_ventilation_mode]

# ---------------------------------------------------------------------------
# Security tools
# ---------------------------------------------------------------------------


@tool
async def check_cameras(zone: str) -> str:
    """Check security camera status for a building zone.

    Args:
        zone: Building zone to check cameras.
    """
    roll = random.random()
    if roll < 0.20:
        detail = f"MOTION DETECTED — {random.choice(['person', 'person (unrecognized)', 'vehicle'])} — confidence {random.randint(75, 98)}%"
    elif roll < 0.25:
        detail = "OFFLINE — camera not responding"
    else:
        detail = "CLEAR — no motion detected"
    return f"Camera Sweep — {zone}\n  {detail}\n  Cameras online: {random.randint(2, 4)}/4"


@tool
async def lock_unlock_door(door_id: str, action: str) -> str:
    """Lock or unlock a building door.

    Args:
        door_id: Door identifier (e.g., MAIN-01, SR-01, FL2-EAST).
        action: Either 'lock' or 'unlock'.
    """
    return f"Door {door_id}: {action.upper()} confirmed. Status: {'SECURED' if action == 'lock' else 'ACCESSIBLE'}"


@tool
async def trigger_alarm(zone: str, alarm_type: str) -> str:
    """Trigger a building alarm.

    Args:
        zone: Building zone for the alarm.
        alarm_type: Type of alarm (fire, intrusion, evacuation).
    """
    return f"ALARM ACTIVATED: {alarm_type.upper()} in {zone}. Security team notified. Strobe/siren: ACTIVE"


@tool
async def log_access_event(person: str, door: str, action: str) -> str:
    """Log a building access event.

    Args:
        person: Name or ID of the person.
        door: Door identifier.
        action: Access action (entry, exit, denied, tailgate).
    """
    return f"Access Logged: {person} — {action.upper()} at {door} [{datetime.now().strftime('%H:%M:%S')}]"


security_tools = [check_cameras, lock_unlock_door, trigger_alarm, log_access_event]

# ---------------------------------------------------------------------------
# Energy tools
# ---------------------------------------------------------------------------


@tool
async def read_power_meters(zone: str) -> str:
    """Read power consumption for a building zone.

    Args:
        zone: Building zone to read meters.
    """
    kwh = random.uniform(5.0, 45.0)
    draw = random.uniform(1.5, 15.0)
    return f"Power — {zone}\n  Consumption: {kwh:.1f} kWh\n  Current draw: {draw:.1f} kW"


@tool
async def toggle_lighting_zone(zone: str, brightness_pct: int) -> str:
    """Adjust lighting level for a building zone.

    Args:
        zone: Building zone to adjust.
        brightness_pct: Brightness percentage (0-100).
    """
    return f"Lighting — {zone}: {brightness_pct}%. Fixtures: {random.randint(8, 24)}"


@tool
async def check_solar_output() -> str:
    """Check rooftop solar panel output and battery status."""
    solar_kw = random.uniform(2.0, 18.0)
    battery_pct = random.randint(20, 95)
    return f"Solar: {solar_kw:.1f} kW output. Battery: {battery_pct}%"


@tool
async def set_power_mode(mode: str) -> str:
    """Set building-wide power management mode.

    Args:
        mode: Power mode (normal, eco, emergency).
    """
    effects = {
        "normal": "All systems standard. No restrictions.",
        "eco": "Non-essential reduced. HVAC setback. Lighting 60%.",
        "emergency": "Critical systems only. Emergency lighting ON.",
    }
    return f"Power Mode: {mode.upper()}. {effects.get(mode, 'Standard.')}"


energy_tools = [read_power_meters, toggle_lighting_zone, check_solar_output, set_power_mode]

# ---------------------------------------------------------------------------
# Maintenance tools
# ---------------------------------------------------------------------------


@tool
async def create_work_order(title: str, priority: str, location: str) -> str:
    """Create a maintenance work order.

    Args:
        title: Description of the work needed.
        priority: Priority level (critical, high, medium, low).
        location: Building location for the work.
    """
    wo_id = f"WO-{datetime.now().strftime('%m%d')}-{random.randint(100, 999)}"
    return f"Work Order {wo_id}: {title} [{priority.upper()}] at {location}"


@tool
async def check_equipment_status(equipment: str) -> str:
    """Check the operational status of building equipment.

    Args:
        equipment: Equipment to check (hvac-unit-1, hvac-unit-2, elevator-1, elevator-2, fire-suppression, generator).
    """
    statuses = {
        "hvac-unit-1": ("OPERATIONAL", "Filter life: 62%"),
        "hvac-unit-2": (
            random.choice(["OPERATIONAL", "NEEDS SERVICE"]),
            f"Refrigerant: {random.choice(['OK', 'LOW'])}",
        ),
        "elevator-1": ("OPERATIONAL", f"Trips today: {random.randint(20, 200)}"),
        "elevator-2": (
            random.choice(["OPERATIONAL", "MINOR FAULT"]),
            f"Door sensor: {random.choice(['OK', 'NEEDS ADJUSTMENT'])}",
        ),
        "fire-suppression": ("OPERATIONAL", f"Pressure: {random.randint(90, 100)}% nominal"),
        "generator": ("STANDBY", f"Fuel: {random.randint(60, 100)}%"),
    }
    status, detail = statuses.get(equipment, ("UNKNOWN", f"'{equipment}' not in registry"))
    return f"Equipment {equipment}: {status}. {detail}"


@tool
async def schedule_inspection(system: str, date: str) -> str:
    """Schedule an inspection for a building system.

    Args:
        system: System to inspect (hvac, electrical, plumbing, fire-safety, elevators).
        date: Target date for inspection.
    """
    return f"Inspection scheduled: {system} on {date}. Inspector: {random.choice(['BuildingCert LLC', 'SafetyFirst Inc.'])}"


@tool
async def order_parts(part_name: str, quantity: int) -> str:
    """Order replacement parts for building equipment.

    Args:
        part_name: Name or part number of the component.
        quantity: Number of units to order.
    """
    cost = random.uniform(25.0, 500.0) * quantity
    return f"Ordered {quantity}x {part_name}. Est. cost: ${cost:.2f}. Delivery: {random.randint(1, 5)} days"


maintenance_tools = [create_work_order, check_equipment_status, schedule_inspection, order_parts]

# ---------------------------------------------------------------------------
# Actor factories
# ---------------------------------------------------------------------------


def make_hvac(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "hvac",
        prompt=(
            "You are the HVAC Controller for a commercial building.\n"
            "You manage climate control across 5 zones: lobby, office-floor-2, "
            "office-floor-3, server-room, parking.\n\n"
            "Your responsibilities:\n"
            "1. Monitor temperatures and keep all zones in the 68-74\u00b0F comfort range\n"
            "2. Adjust thermostats for any out-of-range zones\n"
            "3. Monitor air quality and adjust ventilation as needed\n"
            "4. Coordinate with other building systems when needed\n\n"
            "Use discover_agents and delegate_to if you need help from security, "
            "energy, or maintenance. Always report your findings clearly."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=hvac_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000), LoopDetector(repeat_threshold=3)],
    )


def make_security(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "security",
        prompt=(
            "You are the Security Manager for a commercial building.\n"
            "You oversee access control, surveillance, and alarm systems.\n\n"
            "Your responsibilities:\n"
            "1. Monitor cameras across all zones for threats\n"
            "2. Manage door locks and access control\n"
            "3. Trigger alarms when security events are detected\n"
            "4. Log all access events for audit\n\n"
            "For security incidents, coordinate with other building systems: "
            "use discover_agents and delegate_to to ask energy for emergency "
            "lighting or maintenance for equipment checks. Be thorough and "
            "decisive during emergencies."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=security_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000), LoopDetector(repeat_threshold=3)],
    )


def make_energy(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "energy",
        prompt=(
            "You are the Energy Manager for a commercial building.\n"
            "You control power distribution, lighting, and solar systems.\n\n"
            "Your responsibilities:\n"
            "1. Monitor power consumption across all zones\n"
            "2. Optimize lighting based on occupancy and time of day\n"
            "3. Manage solar panel output and battery storage\n"
            "4. Switch power modes (normal/eco/emergency) as needed\n\n"
            "Use discover_agents and delegate_to to coordinate with HVAC, "
            "security, and maintenance when energy decisions affect their systems."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=energy_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000), LoopDetector(repeat_threshold=3)],
    )


def make_maintenance(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "maintenance",
        prompt=(
            "You are the Maintenance Manager for a commercial building.\n"
            "You oversee repairs, inspections, and equipment health.\n\n"
            "Your responsibilities:\n"
            "1. Check equipment status and create work orders for issues\n"
            "2. Schedule inspections for building systems\n"
            "3. Order replacement parts when needed\n"
            "4. Coordinate emergency repairs with other building systems\n\n"
            "For emergencies, use discover_agents and delegate_to to coordinate "
            "with HVAC (for climate equipment), security (for safety checks), "
            "and energy (for power-related issues). Prioritize critical systems."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=maintenance_tools,
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000), LoopDetector(repeat_threshold=3)],
    )


# ---------------------------------------------------------------------------
# Hub stream logger
# ---------------------------------------------------------------------------


def subscribe_hub_logging(hub: Hub, label: str = "HUB") -> None:
    """Subscribe to hub.stream for live event logging."""

    async def _on_event(event: object) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if isinstance(event, SchedulerTriggerFired):
            print(f"\n  {DIM}{ts}{RESET}  {CYAN}{BOLD}=== SCHEDULER: [{event.target.upper()}] triggered ==={RESET}")
            preview = event.task[:100].replace("\n", " ")
            print(f"  {DIM}{' ' * 12}       {CYAN}{preview}{RESET}")
        elif isinstance(event, DelegationRequest):
            print(
                f"  {DIM}{ts}{RESET}  "
                f"{MAGENTA}{BOLD}{label}{RESET}  "
                f"{MAGENTA}{event.source.upper()} -> {event.target.upper()}{RESET}"
            )
            preview = event.task[:120].replace("\n", " ")
            print(f"  {DIM}{' ' * 12}       {preview}{'...' if len(event.task) > 120 else ''}{RESET}")
        elif isinstance(event, DelegationResult):
            print(
                f"  {DIM}{ts}{RESET}  "
                f"{GREEN}{BOLD}{label}{RESET}  "
                f"{GREEN}{event.target.upper()} done -> {event.source.upper()}{RESET}"
            )
        elif isinstance(event, Signal):
            print(f"  {DIM}{ts}{RESET}  {RED}{BOLD}ALERT [{event.severity.upper()}]{RESET} {RED}{event.message}{RESET}")

    hub.stream.subscribe(_on_event)
