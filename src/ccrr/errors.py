from enum import StrEnum


class Status(StrEnum):
    SUCCESS = "Success"
    INPUT = "Input/Profile Error"
    CAPACITY = "Insufficient Carrier Capacity"
    NONCE = "Nonce State Error"
    EXTRACTION = "Extraction Failure"
    NO_CREDENTIAL = "Credential Unavailable"
    AUTH = "Authentication Failure"
    DECODE = "Entropy/Schema Decode Failure"
    NUMERICAL = "Numerical Recovery Failure"


class CCRRError(Exception):
    def __init__(self, status: Status, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def require(condition, detail, status=Status.INPUT):
    if not condition:
        raise CCRRError(status, detail)
