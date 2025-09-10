#!/usr/bin/env python3
"""
Automated Escalation System for Trading Framework
Handles critical errors and notifications between agents
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class EscalationSystem:
    """Manages escalation notifications and issue tracking."""
    
    def __init__(self):
        self.escalations_dir = Path("docs/escalations")
        self.escalations_dir.mkdir(exist_ok=True)
        
    def escalate_to_builder(
        self, 
        run_id: str, 
        issue_type: str, 
        severity: str,
        description: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Escalate critical issues to the Builder agent.
        
        Args:
            run_id: Run ID where issue occurred
            issue_type: Type of issue (ACCOUNTING_ERROR, VALIDATION_FAILURE, etc.)
            severity: P0 (Critical), P1 (Major), P2 (Minor)
            description: Human-readable description of the issue
            data: Additional diagnostic data
        """
        escalation_id = f"ESC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"
        
        escalation = {
            "escalation_id": escalation_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "issue_type": issue_type,
            "severity": severity,
            "description": description,
            "status": "OPEN",
            "assigned_to": "builder",
            "reporter": "evaluator",
            "data": data or {}
        }
        
        # Save escalation file
        escalation_file = self.escalations_dir / f"{escalation_id}.json"
        with open(escalation_file, 'w') as f:
            json.dump(escalation, f, indent=2)
        
        # Update escalation registry
        self._update_registry(escalation)
        
        # Print notification (in real system, would send email/Slack/etc.)
        print(f"🚨 ESCALATION {severity}: {issue_type}")
        print(f"   ID: {escalation_id}")
        print(f"   Run: {run_id}")
        print(f"   Description: {description}")
        print(f"   Assigned to: Builder")
        
        return escalation_id
    
    def escalate_accounting_error(self, run_id: str, final_equity: float, expected_equity: float, discrepancy_pct: float):
        """Specific escalation for accounting errors."""
        description = f"Critical accounting discrepancy detected. Final equity: ${final_equity:,.2f}, Expected: ${expected_equity:,.2f}, Discrepancy: {discrepancy_pct:.2f}%"
        
        data = {
            "final_equity": final_equity,
            "expected_equity": expected_equity,
            "discrepancy_pct": discrepancy_pct,
            "investigation_required": [
                "Review portfolio_manager.py total_equity calculation",
                "Check for unrealized P&L inclusion errors",
                "Validate fee calculation accuracy",
                "Examine open position handling"
            ]
        }
        
        return self.escalate_to_builder(
            run_id=run_id,
            issue_type="ACCOUNTING_ERROR",
            severity="P0",
            description=description,
            data=data
        )
    
    def escalate_validation_failure(self, run_id: str, validation_type: str, details: str):
        """Specific escalation for validation failures."""
        description = f"Validation failure in {validation_type}: {details}"
        
        return self.escalate_to_builder(
            run_id=run_id,
            issue_type="VALIDATION_FAILURE", 
            severity="P1",
            description=description,
            data={"validation_type": validation_type, "details": details}
        )
    
    def _update_registry(self, escalation: Dict[str, Any]):
        """Update the escalation registry CSV."""
        registry_path = self.escalations_dir / "escalation_registry.csv"
        
        # Create header if file doesn't exist
        if not registry_path.exists():
            with open(registry_path, 'w') as f:
                f.write("escalation_id,timestamp,run_id,issue_type,severity,status,assigned_to,reporter,description\n")
        
        # Append escalation record
        with open(registry_path, 'a') as f:
            f.write(f"{escalation['escalation_id']},{escalation['timestamp']},{escalation['run_id']},{escalation['issue_type']},{escalation['severity']},{escalation['status']},{escalation['assigned_to']},{escalation['reporter']},\"{escalation['description']}\"\n")
    
    def get_open_escalations(self) -> list:
        """Get all open escalations."""
        escalations = []
        for escalation_file in self.escalations_dir.glob("ESC_*.json"):
            with open(escalation_file, 'r') as f:
                escalation = json.load(f)
                if escalation.get('status') == 'OPEN':
                    escalations.append(escalation)
        return escalations
    
    def close_escalation(self, escalation_id: str, resolution: str):
        """Mark an escalation as resolved."""
        escalation_file = self.escalations_dir / f"{escalation_id}.json"
        if escalation_file.exists():
            with open(escalation_file, 'r') as f:
                escalation = json.load(f)
            
            escalation['status'] = 'RESOLVED'
            escalation['resolution'] = resolution
            escalation['resolved_timestamp'] = datetime.utcnow().isoformat() + "Z"
            
            with open(escalation_file, 'w') as f:
                json.dump(escalation, f, indent=2)
            
            print(f"✅ Escalation {escalation_id} marked as RESOLVED")
            print(f"   Resolution: {resolution}")

# Global escalation system instance
escalation_system = EscalationSystem()

def escalate_accounting_error(run_id: str, final_equity: float, expected_equity: float, discrepancy_pct: float) -> str:
    """Quick function for accounting error escalation."""
    return escalation_system.escalate_accounting_error(run_id, final_equity, expected_equity, discrepancy_pct)

def escalate_validation_failure(run_id: str, validation_type: str, details: str) -> str:
    """Quick function for validation failure escalation."""
    return escalation_system.escalate_validation_failure(run_id, validation_type, details)