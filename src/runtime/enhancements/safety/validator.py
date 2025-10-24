"""Safety and security validation system for OM1."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import json


class SafetyLevel(Enum):
    """Safety levels for different operations."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGEROUS = "DANGEROUS"
    FORBIDDEN = "FORBIDDEN"


class ValidationResult(Enum):
    """Validation results."""
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass
class SafetyRule:
    """A safety rule definition."""
    name: str
    pattern: str
    safety_level: SafetyLevel
    description: str
    action_type: Optional[str] = None
    max_speed: Optional[float] = None
    max_distance: Optional[float] = None
    requires_human: bool = False


@dataclass
class SafetyViolation:
    """A safety violation."""
    rule_name: str
    action: str
    safety_level: SafetyLevel
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class ActionSafetyValidator:
    """Validator for robot action safety."""
    
    def __init__(self):
        self.rules: List[SafetyRule] = []
        self.violations: List[SafetyViolation] = []
        self._logger = logging.getLogger("action_safety_validator")
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default safety rules."""
        # Movement safety rules
        self.rules.extend([
            SafetyRule(
                name="high_speed_movement",
                pattern=r"run|sprint|fast",
                safety_level=SafetyLevel.CAUTION,
                description="High-speed movement requires caution",
                action_type="movement",
                max_speed=2.0
            ),
            SafetyRule(
                name="jumping",
                pattern=r"jump|leap|hop",
                safety_level=SafetyLevel.DANGEROUS,
                description="Jumping actions are dangerous",
                action_type="movement",
                requires_human=True
            ),
            SafetyRule(
                name="backflip",
                pattern=r"backflip|flip",
                safety_level=SafetyLevel.FORBIDDEN,
                description="Backflips are forbidden for safety",
                action_type="movement"
            ),
            SafetyRule(
                name="long_distance_movement",
                pattern=r"go to|move to|walk to",
                safety_level=SafetyLevel.CAUTION,
                description="Long distance movement requires caution",
                action_type="movement",
                max_distance=10.0
            ),
            SafetyRule(
                name="stair_climbing",
                pattern=r"climb|stairs|up|down",
                safety_level=SafetyLevel.DANGEROUS,
                description="Stair climbing is dangerous",
                action_type="movement",
                requires_human=True
            )
        ])
        
        # Interaction safety rules
        self.rules.extend([
            SafetyRule(
                name="human_interaction",
                pattern=r"touch|grab|hold|pick up",
                safety_level=SafetyLevel.CAUTION,
                description="Human interaction requires caution",
                action_type="interaction",
                requires_human=True
            ),
            SafetyRule(
                name="object_manipulation",
                pattern=r"pick up|grab|throw|drop",
                safety_level=SafetyLevel.CAUTION,
                description="Object manipulation requires caution",
                action_type="manipulation"
            )
        ])
        
        # System safety rules
        self.rules.extend([
            SafetyRule(
                name="system_shutdown",
                pattern=r"shutdown|power off|turn off",
                safety_level=SafetyLevel.DANGEROUS,
                description="System shutdown is dangerous",
                action_type="system",
                requires_human=True
            ),
            SafetyRule(
                name="configuration_change",
                pattern=r"change config|modify settings|update",
                safety_level=SafetyLevel.CAUTION,
                description="Configuration changes require caution",
                action_type="system"
            )
        ])
    
    def add_rule(self, rule: SafetyRule):
        """Add a custom safety rule."""
        self.rules.append(rule)
        self._logger.info(f"Added safety rule: {rule.name}")
    
    def validate_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a robot action for safety.
        
        Parameters
        ----------
        action : str
            The action to validate
        context : Optional[Dict[str, Any]]
            Additional context for validation
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        context = context or {}
        action_lower = action.lower()
        
        # Check against all rules
        for rule in self.rules:
            if re.search(rule.pattern, action_lower, re.IGNORECASE):
                # Check if action type matches
                if rule.action_type and context.get("action_type") != rule.action_type:
                    continue
                
                # Record violation
                violation = SafetyViolation(
                    rule_name=rule.name,
                    action=action,
                    safety_level=rule.safety_level,
                    message=rule.description,
                    timestamp=time.time(),
                    details={
                        "pattern": rule.pattern,
                        "context": context
                    }
                )
                self.violations.append(violation)
                
                # Determine validation result based on safety level
                if rule.safety_level == SafetyLevel.FORBIDDEN:
                    self._logger.warning(f"FORBIDDEN action blocked: {action} (rule: {rule.name})")
                    return ValidationResult.BLOCKED
                elif rule.safety_level == SafetyLevel.DANGEROUS:
                    if rule.requires_human and not context.get("human_approval", False):
                        self._logger.warning(f"DANGEROUS action requires approval: {action} (rule: {rule.name})")
                        return ValidationResult.REQUIRES_APPROVAL
                    else:
                        self._logger.warning(f"DANGEROUS action allowed with approval: {action} (rule: {rule.name})")
                        return ValidationResult.ALLOWED
                elif rule.safety_level == SafetyLevel.CAUTION:
                    self._logger.info(f"CAUTION action: {action} (rule: {rule.name})")
                    return ValidationResult.ALLOWED
        
        # No violations found
        return ValidationResult.ALLOWED
    
    def get_violations(self, limit: Optional[int] = None) -> List[SafetyViolation]:
        """Get recent safety violations."""
        violations = self.violations
        if limit:
            violations = violations[-limit:]
        return violations
    
    def get_violations_by_level(self, safety_level: SafetyLevel) -> List[SafetyViolation]:
        """Get violations by safety level."""
        return [v for v in self.violations if v.safety_level == safety_level]


class InputSanitizer:
    """Input sanitization for security."""
    
    def __init__(self):
        self._logger = logging.getLogger("input_sanitizer")
        
        # Dangerous patterns
        self.dangerous_patterns = [
            r"<script.*?>.*?</script>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"on\w+\s*=",  # Event handlers
            r"eval\s*\(",  # Eval function
            r"exec\s*\(",  # Exec function
            r"system\s*\(",  # System function
            r"shell\s*\(",  # Shell function
            r"rm\s+-rf",  # Dangerous rm command
            r"sudo\s+",  # Sudo commands
            r"chmod\s+777",  # Dangerous chmod
            r"passwd\s+",  # Password changes
            r"useradd\s+",  # User creation
            r"usermod\s+",  # User modification
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def sanitize(self, input_text: str) -> Dict[str, Any]:
        """
        Sanitize input text for security.
        
        Parameters
        ----------
        input_text : str
            Input text to sanitize
            
        Returns
        -------
        Dict[str, Any]
            Sanitization result with cleaned text and security info
        """
        result = {
            "original": input_text,
            "sanitized": input_text,
            "is_safe": True,
            "threats_detected": [],
            "sanitization_applied": False
        }
        
        # Check for dangerous patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(input_text):
                threat = {
                    "pattern_index": i,
                    "pattern": self.dangerous_patterns[i],
                    "match": pattern.search(input_text).group()
                }
                result["threats_detected"].append(threat)
                result["is_safe"] = False
        
        # If threats detected, sanitize
        if not result["is_safe"]:
            result["sanitized"] = self._apply_sanitization(input_text)
            result["sanitization_applied"] = True
            self._logger.warning(f"Input sanitized due to threats: {len(result['threats_detected'])} threats detected")
        
        return result
    
    def _apply_sanitization(self, text: str) -> str:
        """Apply sanitization to remove dangerous content."""
        # Remove script tags
        text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove event handlers
        text = re.sub(r"on\w+\s*=\s*[^>]*", "", text, flags=re.IGNORECASE)
        
        # Remove dangerous function calls
        text = re.sub(r"(eval|exec|system|shell)\s*\(", "blocked(", text, flags=re.IGNORECASE)
        
        # Remove dangerous commands
        dangerous_commands = ["rm -rf", "sudo", "chmod 777", "passwd", "useradd", "usermod"]
        for cmd in dangerous_commands:
            text = text.replace(cmd, "[BLOCKED]")
        
        # Remove JavaScript URLs
        text = re.sub(r"javascript:", "blocked:", text, flags=re.IGNORECASE)
        
        return text.strip()


class SecurityAuditor:
    """Security auditor for system-wide security checks."""
    
    def __init__(self):
        self._logger = logging.getLogger("security_auditor")
        self.audit_log: List[Dict[str, Any]] = []
        self.suspicious_activities: List[Dict[str, Any]] = []
    
    def audit_action(self, action: str, user: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Audit a user action for security."""
        context = context or {}
        
        audit_entry = {
            "timestamp": time.time(),
            "action": action,
            "user": user,
            "context": context,
            "risk_level": self._assess_risk(action, context)
        }
        
        self.audit_log.append(audit_entry)
        
        # Check for suspicious patterns
        if self._is_suspicious(action, context):
            self.suspicious_activities.append(audit_entry)
            self._logger.warning(f"Suspicious activity detected: {action} by {user}")
    
    def _assess_risk(self, action: str, context: Dict[str, Any]) -> str:
        """Assess risk level of an action."""
        high_risk_keywords = ["delete", "remove", "shutdown", "restart", "config", "admin"]
        medium_risk_keywords = ["modify", "change", "update", "install"]
        
        action_lower = action.lower()
        
        if any(keyword in action_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _is_suspicious(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if an action is suspicious."""
        # Rapid successive actions
        recent_actions = [entry for entry in self.audit_log[-10:] if time.time() - entry["timestamp"] < 60]
        if len(recent_actions) > 5:
            return True
        
        # Unusual time patterns (if we had user history)
        # This would require user behavior analysis
        
        # High-risk actions from unknown users
        if context.get("user_type") == "unknown" and self._assess_risk(action, context) == "HIGH":
            return True
        
        return False
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log
    
    def get_suspicious_activities(self) -> List[Dict[str, Any]]:
        """Get suspicious activities."""
        return self.suspicious_activities


class SafetyManager:
    """Central safety and security management system."""
    
    def __init__(self):
        self.action_validator = ActionSafetyValidator()
        self.input_sanitizer = InputSanitizer()
        self.security_auditor = SecurityAuditor()
        self._logger = logging.getLogger("safety_manager")
        
        # Emergency stop state
        self.emergency_stop_active = False
        self.emergency_stop_reason = None
    
    def validate_and_sanitize_input(self, input_text: str, user: Optional[str] = None) -> Dict[str, Any]:
        """Validate and sanitize input text."""
        # Sanitize input
        sanitization_result = self.input_sanitizer.sanitize(input_text)
        
        # Audit the action
        self.security_auditor.audit_action(
            f"input: {input_text[:100]}...",
            user,
            {"sanitized": sanitization_result["sanitization_applied"]}
        )
        
        return sanitization_result
    
    def validate_action(self, action: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a robot action."""
        context = context or {}
        
        # Check emergency stop
        if self.emergency_stop_active:
            self._logger.warning(f"Action blocked due to emergency stop: {action}")
            return ValidationResult.BLOCKED
        
        # Validate action safety
        result = self.action_validator.validate_action(action, context)
        
        # Audit the action
        self.security_auditor.audit_action(action, context.get("user"), context)
        
        return result
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        self.emergency_stop_active = True
        self.emergency_stop_reason = reason
        self._logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Log as critical security event
        self.security_auditor.audit_action(
            "EMERGENCY_STOP",
            "system",
            {"reason": reason, "timestamp": time.time()}
        )
    
    def clear_emergency_stop(self, reason: str):
        """Clear emergency stop."""
        self.emergency_stop_active = False
        self.emergency_stop_reason = None
        self._logger.info(f"Emergency stop cleared: {reason}")
        
        # Log the clearance
        self.security_auditor.audit_action(
            "EMERGENCY_STOP_CLEARED",
            "system",
            {"reason": reason, "timestamp": time.time()}
        )
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "emergency_stop_reason": self.emergency_stop_reason,
            "recent_violations": len(self.action_validator.get_violations(10)),
            "suspicious_activities": len(self.security_auditor.get_suspicious_activities()),
            "total_audit_entries": len(self.security_auditor.get_audit_log())
        }
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        return {
            "safety_status": self.get_safety_status(),
            "recent_violations": [
                {
                    "rule": v.rule_name,
                    "action": v.action,
                    "level": v.safety_level.value,
                    "message": v.message,
                    "timestamp": v.timestamp
                }
                for v in self.action_validator.get_violations(20)
            ],
            "suspicious_activities": self.security_auditor.get_suspicious_activities()[-10:],
            "audit_summary": {
                "total_entries": len(self.security_auditor.get_audit_log()),
                "high_risk_actions": len([
                    entry for entry in self.security_auditor.get_audit_log()
                    if entry["risk_level"] == "HIGH"
                ]),
                "recent_activities": self.security_auditor.get_audit_log(50)
            }
        }


# Global safety manager instance
safety_manager = SafetyManager()
