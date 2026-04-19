def get_guidelines(risk: float) -> list:
    if risk >= 0.75:
        return [
            "Send SMS reminder 24 hours before appointment",
            "Call patient for confirmation",
            "Offer easy rescheduling option",
            "Flag for care coordinator follow-up"
        ]
    elif risk >= 0.5:
        return [
            "Send SMS reminder",
            "Email notification",
            "Allow rescheduling"
        ]
    else:
        return [
            "Standard reminder",
            "No special intervention needed"
        ]
