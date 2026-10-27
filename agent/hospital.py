"""Loads hospital data and renders it as context for the LLM."""
import json
import os


class HospitalData:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.raw = json.load(f)

    @property
    def name(self) -> str:
        return self.raw.get("hospital_name", "the hospital")

    @property
    def front_desk(self) -> str:
        return self.raw.get("front_desk_number", "the front desk")

    @property
    def emergency(self) -> str:
        return self.raw.get("emergency_number", "the emergency line")

    def context_text(self) -> str:
        r = self.raw
        lines = [
            f"Hospital name: {r.get('hospital_name', 'Unknown')}",
            f"OPD hours: {r.get('opd_opening', 'N/A')} to {r.get('opd_closing', 'N/A')}",
            f"Front desk number: {r.get('front_desk_number', 'N/A')}",
            f"Emergency number: {r.get('emergency_number', 'N/A')}",
            "",
            "Doctors (on_duty_today means the doctor is in the hospital right now):",
        ]
        for d in r.get("doctors", []):
            lines.append(
                f"- {d['name']}, {d.get('specialty', 'doctor')}. "
                f"OPD days: {', '.join(d.get('opd_days', []))}. "
                f"OPD timings: {d.get('opd_timings', 'N/A')}. "
                f"Chamber: {d.get('chamber', 'N/A')}. "
                f"On duty today: {'yes' if d.get('on_duty_today') else 'no'}"
            )
        lines.append("")
        lines.append("Bed availability (available / total):")
        for bed_type, info in r.get("beds", {}).items():
            lines.append(f"- {bed_type}: {info.get('available', 0)} of {info.get('total', 0)} available")
        return "\n".join(lines)

    def system_prompt(self) -> str:
        return (
            f"You are the friendly voice reception assistant for {self.name}. "
            "A caller is talking to you on the phone. "
            "Answer ONLY using the hospital information below.\n"
            "Rules:\n"
            "- Keep every answer under 30 words, in plain spoken language. "
            "No markdown, no lists, no emojis, no symbols.\n"
            "- If asked whether a doctor is in, use the on-duty-today field. "
            "If the doctor is not in today, say so and mention their usual OPD days.\n"
            "- If asked about beds, state how many are available of each type.\n"
            "- If the caller asks something you cannot answer from the data, "
            f"say you do not have that information and suggest calling the front desk at {self.front_desk}.\n"
            f"- For any emergency, tell the caller to call the emergency number {self.emergency} immediately.\n"
            "- If the caller says goodbye, thanks, or the conversation is finished, "
            "reply with exactly [HANGUP] and nothing else.\n"
            "\nHospital information:\n"
            + self.context_text()
        )