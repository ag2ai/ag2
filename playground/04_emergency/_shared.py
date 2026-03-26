"""Shared tools, actors, and logging for the 04_emergency distributed demo."""

import random
from datetime import datetime

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    Actor,
    DelegationRequest,
    DelegationResult,
    Hub,
    LoopDetector,
    Signal,
    TokenMonitor,
)
from autogen.beta.tools.final import tool

# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

PORTS = {"dispatch": 8900, "medical": 8901, "police": 8902, "fire": 8903}

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

# ---------------------------------------------------------------------------
# Dispatch tools
# ---------------------------------------------------------------------------


@tool
async def log_emergency(caller_name: str, location: str, description: str, severity: str = "unknown") -> str:
    """Log an incoming emergency call and create an incident record.

    Args:
        caller_name: Name of the caller.
        location: Location of the emergency.
        description: Description of what happened.
        severity: Severity assessment (critical/serious/moderate/minor).
    """
    iid = f"INC-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    return (
        f"Incident {iid} created.\n  Caller: {caller_name}\n  Location: {location}\n"
        f"  Description: {description}\n  Severity: {severity}\n  Time: {datetime.now().strftime('%H:%M:%S')}"
    )


# ---------------------------------------------------------------------------
# EMS tools
# ---------------------------------------------------------------------------


@tool
async def dispatch_ambulance(location: str, priority: str = "high") -> str:
    """Dispatch an ambulance unit to the specified location.

    Args:
        location: Address or location description.
        priority: Dispatch priority (critical/high/medium/low).
    """
    uid = f"AMB-{random.randint(100, 999)}"
    eta = random.randint(4, 10)
    return (
        f"Ambulance {uid} dispatched to {location}.\n"
        f"  Priority: {priority}, ETA: {eta} min\n"
        f"  Crew: 2 paramedics + 1 EMT, Equipment: ALS"
    )


@tool
async def assess_patient(symptoms: str, mechanism_of_injury: str = "") -> str:
    """Perform a field medical assessment based on reported symptoms.

    Args:
        symptoms: Reported symptoms and observations.
        mechanism_of_injury: How the injury occurred.
    """
    return (
        f"Field Assessment:\n"
        f"  Mechanism: {mechanism_of_injury}\n"
        f"  Symptoms: {symptoms}\n"
        f"  Triage: RED (Immediate)\n"
        f"  Vitals: BP 90/60, HR 120, SpO2 92%\n"
        f"  Assessment: Suspected internal bleeding, possible fractures\n"
        f"  Recommendation: Immediate transport to Level 1 trauma center"
    )


@tool
async def update_patient_status(status: str, vitals: str = "") -> str:
    """Update the patient's current medical status during transport.

    Args:
        status: Current patient status.
        vitals: Current vital signs.
    """
    return f"Patient Update [{datetime.now().strftime('%H:%M:%S')}]: {status}. Vitals: {vitals}"


# ---------------------------------------------------------------------------
# Police tools
# ---------------------------------------------------------------------------


@tool
async def dispatch_patrol_unit(location: str, unit_type: str = "patrol") -> str:
    """Dispatch a police unit to the scene.

    Args:
        location: Address or location description.
        unit_type: Type of unit (patrol/traffic/supervisor).
    """
    uid = f"UNIT-{random.randint(10, 99)}"
    eta = random.randint(3, 8)
    return f"Police {unit_type} {uid} dispatched to {location}. ETA: {eta} min, Officers: 2"


@tool
async def setup_traffic_control(location: str, action: str, lanes_affected: str = "all") -> str:
    """Set up traffic control measures at the scene.

    Args:
        location: Where to set up traffic control.
        action: Action to take (detour/lane-closure/full-closure).
        lanes_affected: Which lanes are affected.
    """
    return f"Traffic Control at {location}: {action}, lanes: {lanes_affected}. Barriers deployed, nav advisory issued."


@tool
async def file_incident_report(incident_type: str, details: str) -> str:
    """File a police incident report.

    Args:
        incident_type: Type of incident (accident/crime/hazard).
        details: Description of what was found.
    """
    rid = f"RPT-{random.randint(10000, 99999)}"
    return f"Report {rid} filed: {incident_type}. {details}"


# ---------------------------------------------------------------------------
# Hospital tools
# ---------------------------------------------------------------------------


@tool
async def check_er_capacity(department: str = "trauma") -> str:
    """Check current ER capacity and bed availability.

    Args:
        department: Department to check (trauma/general/pediatric).
    """
    avail = random.randint(1, 5)
    return f"ER [{department.upper()}]: {avail}/12 beds, {min(avail, 3)} trauma bays open. Status: ACCEPTING"


@tool
async def prepare_trauma_bay(patient_info: str, eta_minutes: int = 10) -> str:
    """Prepare a trauma bay for an incoming patient.

    Args:
        patient_info: Patient description and injuries.
        eta_minutes: Expected arrival in minutes.
    """
    bay = f"Trauma Bay {random.randint(1, 4)}"
    return (
        f"{bay} prepared for: {patient_info}\n"
        f"  ETA: {eta_minutes} min. Blood on standby, CT reserved, surgical team notified."
    )


@tool
async def assign_specialist(specialty: str, urgency: str = "stat") -> str:
    """Request a medical specialist.

    Args:
        specialty: Specialty needed (trauma-surgery/orthopedics/neurology).
        urgency: How urgent (stat/urgent/routine).
    """
    names = ["Smith", "Johnson", "Lee", "Park", "Chen"]
    doctor = f"Dr. {names[random.randint(0, 4)]}"
    return f"{doctor} ({specialty}) assigned, urgency: {urgency}, ETA: {random.randint(2, 8)} min"


# ---------------------------------------------------------------------------
# Fire department tools
# ---------------------------------------------------------------------------


@tool
async def dispatch_fire_engine(location: str, engine_type: str = "standard") -> str:
    """Dispatch a fire engine to the specified location.

    Args:
        location: Address or location description.
        engine_type: Type of engine (standard/ladder/hazmat).
    """
    uid = f"ENG-{random.randint(100, 999)}"
    eta = random.randint(4, 12)
    crew = random.randint(4, 6)
    return f"Fire Engine {uid} ({engine_type}) dispatched to {location}.\n  ETA: {eta} min, Crew: {crew} firefighters"


@tool
async def assess_fire_hazard(location: str, situation: str) -> str:
    """Assess fire hazard conditions at the scene.

    Args:
        location: Location of the incident.
        situation: Description of current conditions.
    """
    wind_dir = random.choice(["N", "S", "E", "W", "NE", "SW"])
    wind_speed = random.randint(5, 25)
    structures = random.randint(2, 8)
    return (
        f"Fire Hazard Assessment at {location}:\n"
        f"  Situation: {situation}\n"
        f"  Wind: {wind_dir} at {wind_speed} mph\n"
        f"  Risk Level: {'EXTREME' if wind_speed > 15 else 'HIGH'}\n"
        f"  Nearby structures: {structures} within hazard radius\n"
        f"  Recommendation: Establish {200 if wind_speed > 15 else 100}ft perimeter"
    )


@tool
async def establish_perimeter(location: str, radius_feet: int = 200) -> str:
    """Establish a safety perimeter around the incident scene.

    Args:
        location: Center of the perimeter.
        radius_feet: Perimeter radius in feet.
    """
    civilians = random.randint(10, 50)
    return (
        f"Safety perimeter established at {location}.\n"
        f"  Radius: {radius_feet} ft, Access points: 2 (north, south)\n"
        f"  Evacuation: {civilians} civilians cleared"
    )


# ---------------------------------------------------------------------------
# Actor factories
# ---------------------------------------------------------------------------


def make_dispatch(model: str = "gemini-3.1-pro-preview") -> Actor:
    return Actor(
        "dispatch",
        prompt=(
            "You are a 911 Emergency Dispatch Operator.\n\n"
            "PROTOCOL:\n"
            "1. Log the emergency using log_emergency — assess severity as:\n"
            "   - CRITICAL: life-threatening injuries, trapped persons, active fire/danger\n"
            "   - SERIOUS: significant injuries, major property damage\n"
            "   - MODERATE: minor injuries, localized incident\n"
            "   - MINOR: no injuries, property damage only\n"
            "2. Use discover_agents to see available services\n"
            "3. Delegate to relevant services based on the incident:\n"
            "   - Medical emergencies: delegate to 'ems'\n"
            "   - Traffic/security needs: delegate to 'police'\n"
            "   - Fire/hazmat/rescue: delegate to 'fire' (only if available)\n"
            "   - For CRITICAL/SERIOUS: delegate to ALL relevant services\n"
            "   - For MINOR: delegate only to the single most relevant service\n"
            "4. Include full incident details and severity in EVERY delegation\n"
            "5. Summarize the coordinated response with priority classification"
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[log_emergency],
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )


def make_ems(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "ems",
        prompt=(
            "You are an EMS Coordinator managing pre-hospital emergency care.\n\n"
            "PROTOCOL:\n"
            "1. Dispatch an ambulance using dispatch_ambulance\n"
            "2. Assess the patient using assess_patient\n"
            "3. Delegate to 'hospital' with your initial assessment — include injury\n"
            "   details, triage category, and ETA so they can prepare the trauma bay\n"
            "4. Update patient status using update_patient_status (simulating en-route care)\n"
            "5. Delegate to 'hospital' AGAIN with updated vitals and revised ETA —\n"
            "   this is the live transport update so the receiving team is ready\n"
            "6. Provide a complete status summary\n\n"
            "You MUST delegate to hospital TWICE: once for initial preparation,\n"
            "and once with transport updates before arrival."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[dispatch_ambulance, assess_patient, update_patient_status],
        observers=[LoopDetector(repeat_threshold=3)],
    )


def make_police(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "police",
        prompt=(
            "You are a Police Scene Commander.\n"
            "1. Dispatch a patrol/traffic unit using dispatch_patrol_unit\n"
            "2. Set up traffic control using setup_traffic_control\n"
            "3. File an incident report using file_incident_report\n"
            "4. Report scene status\n\n"
            "Prioritize traffic safety first."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[dispatch_patrol_unit, setup_traffic_control, file_incident_report],
        observers=[LoopDetector(repeat_threshold=3)],
    )


def make_hospital(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "hospital",
        prompt=(
            "You are a Hospital ER Coordinator.\n\n"
            "Handle two types of requests:\n\n"
            "INITIAL ALERT (incoming patient notification):\n"
            "  1. Check ER capacity using check_er_capacity\n"
            "  2. Prepare a trauma bay using prepare_trauma_bay\n"
            "  3. Assign specialists using assign_specialist\n"
            "  4. Report full readiness status\n\n"
            "TRANSPORT UPDATE (en-route status update from EMS):\n"
            "  1. Acknowledge updated vitals and ETA\n"
            "  2. Adjust preparations if needed (reassign specialists, prep equipment)\n"
            "  3. Confirm readiness for arrival\n\n"
            "Check capacity first, then prepare and assign."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[check_er_capacity, prepare_trauma_bay, assign_specialist],
        observers=[TokenMonitor(warn_threshold=10_000, alert_threshold=30_000)],
    )


def make_fire_chief(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "fire",
        prompt=(
            "You are a Fire Department Chief.\n"
            "1. Dispatch a fire engine using dispatch_fire_engine\n"
            "   (use 'ladder' type for structure fires, 'hazmat' for chemical incidents)\n"
            "2. Assess fire hazards using assess_fire_hazard\n"
            "3. Establish safety perimeter using establish_perimeter\n"
            "4. Report scene status and actions taken\n\n"
            "Priority order: life safety, exposure protection, property conservation."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=[dispatch_fire_engine, assess_fire_hazard, establish_perimeter],
        observers=[LoopDetector(repeat_threshold=3)],
    )


# ---------------------------------------------------------------------------
# Hub stream logger — subscribe to hub.stream for colored event output
# ---------------------------------------------------------------------------


def subscribe_hub_logging(hub: Hub, label: str = "HUB") -> None:
    """Subscribe to hub.stream for live delegation/result/signal logging."""

    async def _on_event(event: object) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if isinstance(event, DelegationRequest):
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
