"""
Advanced Security and Data Validation Module for OpenBioGen AI
Provides input sanitization, rate limiting, and security monitoring
"""

import re
import time
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Any = None
    security_level: SecurityLevel = SecurityLevel.LOW

class AdvancedValidator:
    """Comprehensive input validation and sanitization"""
    
    # Gene symbol patterns
    GENE_PATTERN = re.compile(r'^[A-Z][A-Z0-9-]*[A-Z0-9]$|^[A-Z]$')
    
    # Disease name patterns (more permissive but safe)
    DISEASE_PATTERN = re.compile(r'^[a-zA-Z0-9\s\-\'\.(),]+$')
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        re.compile(r'<script.*?>', re.IGNORECASE),
        re.compile(r'javascript:', re.IGNORECASE),
        re.compile(r'on\w+\s*=', re.IGNORECASE),
        re.compile(r'eval\s*\(', re.IGNORECASE),
        re.compile(r'exec\s*\(', re.IGNORECASE),
        re.compile(r'__import__', re.IGNORECASE),
        re.compile(r'\.\./', re.IGNORECASE),
        re.compile(r'[;&|`$]', re.IGNORECASE)
    ]
    
    # Known gene symbols for validation
    KNOWN_GENES = {
        'BRCA1', 'BRCA2', 'TP53', 'APOE', 'CFTR', 'HTT', 'SOD1', 'PSEN1', 'PSEN2',
        'APP', 'LRRK2', 'SNCA', 'PARK2', 'PINK1', 'DJ1', 'VHL', 'APC', 'MLH1',
        'MSH2', 'MSH6', 'PMS2', 'CDKN2A', 'RB1', 'NF1', 'NF2', 'TSC1', 'TSC2'
    }
    
    @classmethod
    def validate_gene_symbol(cls, gene: str) -> ValidationResult:
        """Validate gene symbol with comprehensive checks"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        if not gene or not isinstance(gene, str):
            errors.append("Gene symbol must be a non-empty string")
            return ValidationResult(False, errors, warnings, None, SecurityLevel.HIGH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(gene):
                errors.append("Gene symbol contains potentially dangerous characters")
                security_level = SecurityLevel.CRITICAL
                return ValidationResult(False, errors, warnings, None, security_level)
        
        # Sanitize input
        sanitized_gene = gene.strip().upper()
        
        # Length validation
        if len(sanitized_gene) > 20:
            errors.append("Gene symbol too long (max 20 characters)")
            security_level = SecurityLevel.MEDIUM
        
        if len(sanitized_gene) < 1:
            errors.append("Gene symbol too short")
            security_level = SecurityLevel.MEDIUM
        
        # Pattern validation
        if not cls.GENE_PATTERN.match(sanitized_gene):
            errors.append("Gene symbol format invalid (should contain only letters, numbers, and hyphens)")
            security_level = SecurityLevel.MEDIUM
        
        # Check against known genes
        if sanitized_gene not in cls.KNOWN_GENES:
            warnings.append(f"Gene symbol '{sanitized_gene}' not in known gene database")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_gene, security_level)
    
    @classmethod
    def validate_disease_name(cls, disease: str) -> ValidationResult:
        """Validate disease name with comprehensive checks"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        if not disease or not isinstance(disease, str):
            errors.append("Disease name must be a non-empty string")
            return ValidationResult(False, errors, warnings, None, SecurityLevel.HIGH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(disease):
                errors.append("Disease name contains potentially dangerous characters")
                security_level = SecurityLevel.CRITICAL
                return ValidationResult(False, errors, warnings, None, security_level)
        
        # Sanitize input
        sanitized_disease = disease.strip().lower()
        
        # Length validation
        if len(sanitized_disease) > 100:
            errors.append("Disease name too long (max 100 characters)")
            security_level = SecurityLevel.MEDIUM
        
        if len(sanitized_disease) < 2:
            errors.append("Disease name too short (min 2 characters)")
            security_level = SecurityLevel.MEDIUM
        
        # Pattern validation
        if not cls.DISEASE_PATTERN.match(sanitized_disease):
            errors.append("Disease name contains invalid characters")
            security_level = SecurityLevel.MEDIUM
        
        # Common disease validation
        common_diseases = {
            'cancer', 'diabetes', 'alzheimer', 'parkinson', 'huntington', 'cystic fibrosis',
            'sickle cell', 'thalassemia', 'hemophilia', 'muscular dystrophy'
        }
        
        if not any(common in sanitized_disease for common in common_diseases):
            warnings.append("Disease name not recognized in common disease database")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_disease, security_level)
    
    @classmethod
    def validate_input(cls, input_text: str, max_length: int = 100, min_length: int = 1) -> ValidationResult:
        """Generic input validation for compounds and other text inputs"""
        errors = []
        warnings = []
        security_level = SecurityLevel.LOW
        
        if not input_text or not isinstance(input_text, str):
            errors.append("Input must be a non-empty string")
            return ValidationResult(False, errors, warnings, None, SecurityLevel.HIGH)
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if pattern.search(input_text):
                errors.append("Input contains potentially dangerous characters")
                security_level = SecurityLevel.CRITICAL
                return ValidationResult(False, errors, warnings, None, security_level)
        
        # Sanitize input
        sanitized_input = input_text.strip()
        
        # Length validation
        if len(sanitized_input) > max_length:
            errors.append(f"Input too long (max {max_length} characters)")
            security_level = SecurityLevel.MEDIUM
        
        if len(sanitized_input) < min_length:
            errors.append(f"Input too short (min {min_length} characters)")
            security_level = SecurityLevel.MEDIUM
        
        # Basic pattern validation - allow alphanumeric, spaces, hyphens, and common chemical characters
        safe_pattern = re.compile(r'^[a-zA-Z0-9\s\-\.\(\)\+\,\']+$')
        if not safe_pattern.match(sanitized_input):
            errors.append("Input contains invalid characters")
            security_level = SecurityLevel.MEDIUM
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, sanitized_input, security_level)

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.requests = {}  # IP -> list of timestamps
        self.blocked_ips = {}  # IP -> block_until_timestamp
        self.suspicious_patterns = {}  # IP -> pattern_count
    
    def is_allowed(self, identifier: str, max_requests: int = 100, window_seconds: int = 3600) -> Tuple[bool, str]:
        """Check if request is allowed based on rate limiting"""
        current_time = time.time()
        
        # Check if IP is currently blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                return False, "IP temporarily blocked due to suspicious activity"
            else:
                del self.blocked_ips[identifier]
        
        # Initialize request history for new identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < window_seconds
        ]
        
        # Check rate limit
        if len(self.requests[identifier]) >= max_requests:
            # Block IP for suspicious activity
            self.blocked_ips[identifier] = current_time + (window_seconds * 2)
            return False, f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds"
        
        # Add current request
        self.requests[identifier] = self.requests[identifier] + [current_time]
        return True, "Request allowed"
    
    def detect_suspicious_patterns(self, identifier: str, request_data: Dict[str, Any]) -> bool:
        """Detect suspicious request patterns"""
        if identifier not in self.suspicious_patterns:
            self.suspicious_patterns[identifier] = {"rapid_requests": 0, "invalid_inputs": 0}
        
        patterns = self.suspicious_patterns[identifier]
        
        # Check for rapid consecutive requests
        if identifier in self.requests and len(self.requests[identifier]) > 1:
            recent_requests = self.requests[identifier][-10:]  # Last 10 requests
            if len(recent_requests) >= 5:
                time_diff = recent_requests[-1] - recent_requests[0]
                if time_diff < 10:  # 5 requests in 10 seconds
                    patterns["rapid_requests"] += 1
        
        # Check for invalid input patterns
        if "validation_errors" in request_data and request_data["validation_errors"]:
            patterns["invalid_inputs"] += 1
        
        # Determine if suspicious
        is_suspicious = (
            patterns["rapid_requests"] > 3 or
            patterns["invalid_inputs"] > 10
        )
        
        if is_suspicious:
            self.blocked_ips[identifier] = time.time() + 1800  # Block for 30 minutes
        
        return is_suspicious

class SecurityAuditor:
    """Security event logging and monitoring"""
    
    def __init__(self):
        self.security_events = []
        self.setup_audit_log()
    
    def setup_audit_log(self):
        """Setup security audit logging"""
        os.makedirs("security_logs", exist_ok=True)
        self.audit_file = "security_logs/security_audit.log"
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: SecurityLevel = SecurityLevel.LOW):
        """Log security event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "severity": severity.value,
            "details": details,
            "event_id": secrets.token_hex(8)
        }
        
        self.security_events.append(event)
        
        # Write to audit log
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]
        
        summary = {
            "total_events": len(recent_events),
            "events_by_severity": {},
            "events_by_type": {},
            "high_risk_events": []
        }
        
        for event in recent_events:
            # Count by severity
            severity = event["severity"]
            summary["events_by_severity"][severity] = summary["events_by_severity"].get(severity, 0) + 1
            
            # Count by type
            event_type = event["event_type"]
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
            
            # Collect high-risk events
            if event["severity"] in ["high", "critical"]:
                summary["high_risk_events"].append(event)
        
        return summary

# Global instances
rate_limiter = RateLimiter()
security_auditor = SecurityAuditor()

def secure_input_validation(func):
    """Decorator for secure input validation"""
    def wrapper(*args, **kwargs):
        # Extract potential user inputs
        user_inputs = {}
        if args:
            user_inputs["args"] = args
        if kwargs:
            user_inputs["kwargs"] = kwargs
        
        # Log the request
        security_auditor.log_security_event(
            "function_call",
            {"function": func.__name__, "inputs": str(user_inputs)[:200]},
            SecurityLevel.LOW
        )
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            security_auditor.log_security_event(
                "function_error",
                {"function": func.__name__, "error": str(e)},
                SecurityLevel.MEDIUM
            )
            raise
    
    return wrapper
