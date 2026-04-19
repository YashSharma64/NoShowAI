def get_guidelines(risk: float) -> list:
    if risk >= 0.75:
        return [
            "Send SMS reminder",
            "Call patient",
            "Offer rescheduling"
        ]
    elif risk >= 0.5:
        return [
            "Send reminder",
            "Email notification"
        ]
    else:
        return [
            "Standard reminder"
        ]
