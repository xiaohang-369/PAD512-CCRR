"""Prerecorded point targets; a pass does not imply statistical clinical proof."""


def deployment_gate(report, prior=None):
    reasons = []
    for key, threshold in [
        ("coverage", 0.95),
        ("c_all", 0.95),
        ("m_all", 0.95),
        ("both_rate", 0.95),
        ("c_conditional", 0.99),
        ("m_conditional", 0.99),
        ("both_conditional", 0.99),
        ("psnr", 30),
        ("ssim", 0.95),
        ("fraction_psnr28_ssim90", 0.90),
    ]:
        if report.get(key, 0) < threshold:
            reasons.append(key + " below registered target")
    for label, values in report.get("class_coverage", {}).items():
        if any(v < 0.90 for v in values.values()):
            reasons.append(label + " coverage below .90")
    if len(report.get("class_coverage", {})) != 6:
        reasons.append("Class coverage unavailable")
    if not report.get("clinical_check", {}).get("passed_point"):
        reasons.append("Paired clinical utility target not met")
    privacy = report.get("privacy_checks", {})
    if len(privacy) != 4 or not all(v.get("passed") for v in privacy.values()):
        reasons.append("P1/P2 direction-specific privacy target not met")
    if not report.get("source_privacy", {}).get("passed"):
        reasons.append("P3 source/patient matching privacy target not met")
    return {
        "passed": not reasons,
        "failed_conditions": reasons,
        "scope": "Development point-target screening, not independent final clinical/privacy acceptance",
    }
