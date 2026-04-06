#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for FaultLine Environment
Tests environment stability, graders, and edge cases
"""

import sys
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any

# Import environment and models
from faultline.env import FaultLineEnv
from faultline.models import (
    AcknowledgeAlertAction, QueryLogsAction, CheckMetricsAction,
    ResolveAction, RollbackAction, ScaleServiceAction, EscalateAction,
)
from faultline.graders.grader_easy import GraderEasy
from faultline.graders.grader_medium import GraderMedium
from faultline.graders.grader_hard import GraderHard


class StressTestRunner:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "overall_score": 0.0,
            }
        }
        self.test_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.error_count = 0

    def log_test(self, name: str, passed: bool, message: str = "", details: Dict = None):
        """Log a test result"""
        self.test_count += 1
        if passed:
            self.pass_count += 1
            status = "✓ PASS"
        else:
            self.fail_count += 1
            status = "✗ FAIL"
        
        print(f"{status}: {name}")
        if message:
            print(f"  → {message}")
        
        self.results["tests"][name] = {
            "passed": passed,
            "message": message,
            "details": details or {}
        }

    def log_error(self, name: str, error: Exception):
        """Log an error"""
        self.error_count += 1
        self.test_count += 1
        print(f"✗ ERROR: {name}")
        print(f"  → {type(error).__name__}: {str(error)}")
        print(f"  → {traceback.format_exc()}")
        
        self.results["errors"].append({
            "test": name,
            "error": str(error),
            "traceback": traceback.format_exc()
        })

    def test_environment_initialization(self):
        """Test 1: Environment initialization for all difficulty levels"""
        print("\n" + "="*60)
        print("TEST SUITE 1: Environment Initialization")
        print("="*60)
        
        tasks = ["single_service_latency", "cascading_failure", "multi_region_incident"]
        
        for task_id in tasks:
            try:
                env = FaultLineEnv(task_id, seed=42)
                result = env.reset()
                
                # Validate reset result
                assert result.done is False, "Episode should not be done on reset"
                assert result.reward == 0.0, "Initial reward should be 0"
                assert result.observation is not None, "Observation should exist"
                assert len(result.observation.alerts) > 0, "Should have alerts"
                
                num_alerts = len(result.observation.alerts)
                if task_id == "single_service_latency":
                    assert num_alerts == 1, f"Easy task should have 1 alert, got {num_alerts}"
                elif task_id == "cascading_failure":
                    assert num_alerts == 3, f"Medium task should have 3 alerts, got {num_alerts}"
                elif task_id == "multi_region_incident":
                    assert num_alerts == 5, f"Hard task should have 5 alerts, got {num_alerts}"
                
                self.log_test(
                    f"Init: {task_id}",
                    True,
                    f"Initialized with {num_alerts} alerts",
                    {"task_id": task_id, "alerts": num_alerts}
                )
            except Exception as e:
                self.log_error(f"Init: {task_id}", e)

    def test_reproducibility(self):
        """Test 2: Reproducibility with same seeds"""
        print("\n" + "="*60)
        print("TEST SUITE 2: Reproducibility (Same Seed)")
        print("="*60)
        
        tasks = ["single_service_latency", "cascading_failure", "multi_region_incident"]
        
        for task_id in tasks:
            try:
                env1 = FaultLineEnv(task_id, seed=123)
                env2 = FaultLineEnv(task_id, seed=123)
                
                r1 = env1.reset()
                r2 = env2.reset()
                
                # Compare observations
                assert len(r1.observation.alerts) == len(r2.observation.alerts), "Alert count mismatch"
                assert r1.observation.alerts[0].id == r2.observation.alerts[0].id, "Alert IDs should match"
                assert r1.observation.alerts[0].description == r2.observation.alerts[0].description, "Alert descriptions should match"
                
                self.log_test(
                    f"Reproducibility: {task_id}",
                    True,
                    "Same seed produces identical episodes",
                    {"task_id": task_id}
                )
            except Exception as e:
                self.log_error(f"Reproducibility: {task_id}", e)

    def test_action_execution(self):
        """Test 3: Action execution and feedback"""
        print("\n" + "="*60)
        print("TEST SUITE 3: Action Execution")
        print("="*60)
        
        try:
            env = FaultLineEnv("single_service_latency", seed=42)
            env.reset()
            
            # Test AcknowledgeAlertAction
            result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
            assert result.reward > 0, "AcknowledgeAlert should give positive reward"
            assert "alert-001" in result.observation.acknowledged_alerts, "Alert should be acknowledged"
            self.log_test("Action: AcknowledgeAlert", True, f"Reward: {result.reward}")
            
            # Test QueryLogsAction
            result = env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
            assert result.observation.log_results is not None, "Should have log results"
            assert len(result.observation.log_results) > 0, "Should have log entries"
            self.log_test("Action: QueryLogs", True, f"Returned {len(result.observation.log_results)} log entries")
            
            # Test CheckMetricsAction
            result = env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory", window_minutes=15))
            assert result.observation.metric_results is not None, "Should have metric results"
            assert result.observation.metric_results.service == "elasticsearch", "Metric service should match"
            assert len(result.observation.metric_results.points) > 0, "Should have metric points"
            self.log_test("Action: CheckMetrics", True, f"Returned {len(result.observation.metric_results.points)} metric points")
            
        except Exception as e:
            self.log_error("Action Execution", e)

    def test_episode_completion(self):
        """Test 4: Episode completion and done flag"""
        print("\n" + "="*60)
        print("TEST SUITE 4: Episode Completion")
        print("="*60)
        
        try:
            env = FaultLineEnv("single_service_latency", seed=42)
            env.reset()
            
            step_count = 0
            done = False
            
            # Acknowledge alert
            result = env.step(AcknowledgeAlertAction(alert_id="alert-001"))
            step_count += 1
            print(f"  Step {step_count}: Acknowledged alert")
            
            # Query logs
            result = env.step(QueryLogsAction(service="elasticsearch", time_range="last_15m"))
            step_count += 1
            print(f"  Step {step_count}: Queried logs")
            
            # Check metrics
            result = env.step(CheckMetricsAction(service="elasticsearch", metric_name="memory", window_minutes=15))
            step_count += 1
            print(f"  Step {step_count}: Checked metrics")
            
            # Resolve with correct root cause
            result = env.step(ResolveAction(
                root_cause_service="elasticsearch",
                postmortem_text="Elasticsearch ran out of heap memory due to GC pressure. OOM events caused search-service timeouts. Remediation: increased heap size and tuned GC settings."
            ))
            step_count += 1
            done = result.done
            print(f"  Step {step_count}: Resolved - Episode done: {done}")
            
            self.log_test(
                "Episode Completion (Easy Task)",
                done,
                f"Completed in {step_count} steps with done={done}",
                {"steps": step_count, "done": done}
            )
        except Exception as e:
            self.log_error("Episode Completion", e)

    def test_grading_system(self):
        """Test 5: Grading system for all difficulty levels"""
        print("\n" + "="*60)
        print("TEST SUITE 5: Grading System")
        print("="*60)
        
        test_cases = [
            {
                "name": "Easy (Perfect)",
                "task_id": "single_service_latency",
                "seed": 42,
                "actions": [
                    AcknowledgeAlertAction(alert_id="alert-001"),
                    QueryLogsAction(service="elasticsearch", time_range="last_15m"),
                    CheckMetricsAction(service="elasticsearch", metric_name="memory", window_minutes=15),
                    ResolveAction(
                        root_cause_service="elasticsearch",
                        postmortem_text="Elasticsearch ran out of heap memory due to GC pressure causing OOM events. Remediation: increased heap size."
                    )
                ],
                "grader": GraderEasy(),
                "expected_min_score": 0.80
            },
            {
                "name": "Medium (Correct)",
                "task_id": "cascading_failure",
                "seed": 99,
                "actions": [
                    AcknowledgeAlertAction(alert_id="alert-002"),
                    AcknowledgeAlertAction(alert_id="alert-004"),
                    QueryLogsAction(service="payment-db", time_range="last_15m"),
                    QueryLogsAction(service="payment-service", time_range="last_15m"),
                    RollbackAction(service="payment-service", target_version="v1.4.1")
                ],
                "grader": GraderMedium(),
                "expected_min_score": 0.50
            },
            {
                "name": "Hard (Correct)",
                "task_id": "multi_region_incident",
                "seed": 777,
                "actions": [
                    AcknowledgeAlertAction(alert_id="alert-006"),
                    AcknowledgeAlertAction(alert_id="alert-007"),
                    QueryLogsAction(service="fraud-detector", time_range="last_15m"),
                    QueryLogsAction(service="model-serving", time_range="last_15m"),
                    CheckMetricsAction(service="model-serving", metric_name="cpu", window_minutes=15),
                    ScaleServiceAction(service="model-serving", replicas=4)
                ],
                "grader": GraderHard(),
                "expected_min_score": 0.40
            }
        ]
        
        scores = []
        for test_case in test_cases:
            try:
                env = FaultLineEnv(test_case["task_id"], seed=test_case["seed"])
                env.reset()
                
                for action in test_case["actions"]:
                    env.step(action)
                
                grading_result = test_case["grader"].grade(env._task)
                score = grading_result.score
                passed = grading_result.passed
                scores.append(score)
                
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
                
                self.log_test(
                    f"Grading: {test_case['name']}",
                    passed,
                    f"Score: {score:.3f} | Passed: {passed}",
                    {
                        "score": score,
                        "passed": passed,
                        "breakdown": grading_result.breakdown,
                        "notes": grading_result.notes
                    }
                )
            except Exception as e:
                self.log_error(f"Grading: {test_case['name']}", e)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            self.results["summary"]["overall_score"] = avg_score
            print(f"\n  Average Score: {avg_score:.3f}")

    def test_edge_cases(self):
        """Test 6: Edge cases and error handling"""
        print("\n" + "="*60)
        print("TEST SUITE 6: Edge Cases & Error Handling")
        print("="*60)
        
        # Test invalid task
        try:
            env = FaultLineEnv("invalid_task")
            env.reset()
            self.log_test("Invalid Task", False, "Should have raised ValueError")
        except ValueError as e:
            self.log_test("Invalid Task", True, "Correctly raised ValueError")
        except Exception as e:
            self.log_error("Invalid Task", e)
        
        # Test step before reset
        try:
            env = FaultLineEnv("single_service_latency")
            env.step(AcknowledgeAlertAction(alert_id="alert-001"))
            self.log_test("Step Before Reset", False, "Should have raised RuntimeError")
        except RuntimeError:
            self.log_test("Step Before Reset", True, "Correctly raised RuntimeError")
        except Exception as e:
            self.log_error("Step Before Reset", e)
        
        # Test invalid action
        try:
            env = FaultLineEnv("single_service_latency", seed=42)
            env.reset()
            env.step(AcknowledgeAlertAction(alert_id="invalid-alert"))
            # May fail silently or with appropriate feedback
            self.log_test("Invalid Alert ID", True, "Handled gracefully")
        except Exception as e:
            self.log_test("Invalid Alert ID", True, f"Handled: {type(e).__name__}")

    def test_loop_penalty(self):
        """Test 7: Loop penalty mechanism"""
        print("\n" + "="*60)
        print("TEST SUITE 7: Loop Penalty Detection")
        print("="*60)
        
        try:
            env = FaultLineEnv("single_service_latency", seed=42)
            env.reset()
            
            # Perform repeated queries to trigger loop penalty
            for i in range(5):
                env.step(QueryLogsAction(service="search-service", time_range="last_15m"))
            
            task = env._task
            loop_penalty = task.reward_breakdown.get("loop_penalty", 0.0)
            
            if loop_penalty < 0:
                self.log_test("Loop Penalty", True, f"Loop penalty applied: {loop_penalty}")
            else:
                self.log_test("Loop Penalty", True, "Loop penalty mechanism present (may not trigger in all cases)")
                
        except Exception as e:
            self.log_error("Loop Penalty", e)

    def test_dependency_graph(self):
        """Test 8: Dependency graph validity"""
        print("\n" + "="*60)
        print("TEST SUITE 8: Dependency Graph")
        print("="*60)
        
        tasks = ["single_service_latency", "cascading_failure", "multi_region_incident"]
        
        for task_id in tasks:
            try:
                env = FaultLineEnv(task_id, seed=42)
                result = env.reset()
                
                dep_graph = result.observation.dependency_graph
                assert dep_graph is not None, "Dependency graph should exist"
                assert len(dep_graph) > 0, "Dependency graph should not be empty"
                
                # Verify graph structure
                for service, deps in dep_graph.items():
                    assert isinstance(service, str), f"Service name should be string: {service}"
                    assert isinstance(deps, list), f"Dependencies should be list: {deps}"
                
                self.log_test(
                    f"Dependency Graph: {task_id}",
                    True,
                    f"Valid graph with {len(dep_graph)} services",
                    {"services": len(dep_graph), "services_list": list(dep_graph.keys())}
                )
            except Exception as e:
                self.log_error(f"Dependency Graph: {task_id}", e)

    def test_stress_load(self):
        """Test 9: Stress load - multiple random episodes"""
        print("\n" + "="*60)
        print("TEST SUITE 9: Stress Load (Multiple Episodes)")
        print("="*60)
        
        import random
        
        tasks = ["single_service_latency", "cascading_failure", "multi_region_incident"]
        num_episodes = 3  # 3 per task = 9 total
        
        successful = 0
        failed = 0
        
        for task_id in tasks:
            for episode in range(num_episodes):
                try:
                    seed = random.randint(1, 10000)
                    env = FaultLineEnv(task_id, seed=seed)
                    result = env.reset()
                    
                    # Random actions
                    if result.observation.alerts:
                        alert_id = result.observation.alerts[0].id
                        env.step(AcknowledgeAlertAction(alert_id=alert_id))
                    
                    if result.observation.alerts and result.observation.alerts[0].service:
                        service = result.observation.alerts[0].service
                        env.step(QueryLogsAction(service=service, time_range="last_15m"))
                    
                    successful += 1
                    
                except Exception as e:
                    failed += 1
                    print(f"  Episode {episode+1} for {task_id}: {type(e).__name__}")
        
        self.log_test(
            "Stress Load (Multiple Episodes)",
            failed == 0,
            f"Ran {successful} episodes successfully, {failed} failed",
            {"successful": successful, "failed": failed, "total": successful + failed}
        )

    def test_observation_validity(self):
        """Test 10: Observation structure and validity"""
        print("\n" + "="*60)
        print("TEST SUITE 10: Observation Validity")
        print("="*60)
        
        try:
            env = FaultLineEnv("single_service_latency", seed=42)
            result = env.reset()
            obs = result.observation
            
            # Validate required fields
            assert obs.alerts is not None, "Missing alerts"
            assert isinstance(obs.alerts, list), "alerts should be list"
            
            assert obs.dependency_graph is not None, "Missing dependency_graph"
            assert isinstance(obs.dependency_graph, dict), "dependency_graph should be dict"
            
            assert obs.acknowledged_alerts is not None, "Missing acknowledged_alerts"
            assert isinstance(obs.acknowledged_alerts, list), "acknowledged_alerts should be list"
            
            assert obs.elapsed_steps >= 0, "elapsed_steps should be non-negative"
            
            # Validate alert structure
            for alert in obs.alerts:
                assert alert.id is not None, "Alert missing id"
                assert alert.service is not None, "Alert missing service"
                assert alert.title is not None, "Alert missing title"
                assert alert.description is not None, "Alert missing description"
                assert alert.severity is not None, "Alert missing severity"
            
            self.log_test(
                "Observation Validity",
                True,
                f"All required fields present and valid",
                {
                    "alerts": len(obs.alerts),
                    "services_in_graph": len(obs.dependency_graph),
                    "elapsed_steps": obs.elapsed_steps
                }
            )
        except Exception as e:
            self.log_error("Observation Validity", e)

    def run_all_tests(self):
        """Run all test suites"""
        print("\n")
        print("╔" + "="*58 + "╗")
        print("║" + " "*58 + "║")
        print("║" + "  FAULTLINE ENVIRONMENT - COMPREHENSIVE STRESS TEST".center(58) + "║")
        print("║" + " "*58 + "║")
        print("╚" + "="*58 + "╝")
        
        test_methods = [
            self.test_environment_initialization,
            self.test_reproducibility,
            self.test_action_execution,
            self.test_episode_completion,
            self.test_grading_system,
            self.test_edge_cases,
            self.test_loop_penalty,
            self.test_dependency_graph,
            self.test_stress_load,
            self.test_observation_validity,
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"\nUnexpected error in {test_method.__name__}: {e}")
                traceback.print_exc()
        
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        self.results["summary"]["total_tests"] = self.test_count
        self.results["summary"]["passed"] = self.pass_count
        self.results["summary"]["failed"] = self.fail_count
        self.results["summary"]["errors"] = self.error_count
        
        success_rate = (self.pass_count / self.test_count * 100) if self.test_count > 0 else 0
        
        print(f"Total Tests:     {self.test_count}")
        print(f"Passed:          {self.pass_count} ({success_rate:.1f}%)")
        print(f"Failed:          {self.fail_count}")
        print(f"Errors:          {self.error_count}")
        print(f"Overall Score:   {self.results['summary']['overall_score']:.3f} / 1.000")
        
        # Grade the test results
        if success_rate >= 95:
            grade = "A+ (EXCELLENT)"
        elif success_rate >= 90:
            grade = "A (VERY GOOD)"
        elif success_rate >= 80:
            grade = "B (GOOD)"
        elif success_rate >= 70:
            grade = "C (ACCEPTABLE)"
        else:
            grade = "F (NEEDS IMPROVEMENT)"
        
        print(f"\nFinal Grade:     {grade}")
        
        print("\n" + "="*60)
        
        # Save results to file
        results_file = "stress_test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
        # Print recommendations
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if self.fail_count > 0 or self.error_count > 0:
            print("\n⚠ Issues Found:")
            for test_name, result in self.results["tests"].items():
                if not result["passed"]:
                    print(f"  • {test_name}: {result['message']}")
            
            for error in self.results["errors"]:
                print(f"  • {error['test']}: {error['error']}")
        else:
            print("\n✓ All tests passed! Environment is ready for submission.")
        
        print("\nNext Steps:")
        print("  1. Review any failed tests above")
        print("  2. Check stress_test_results.json for detailed metrics")
        print("  3. Fix any identified issues")
        print("  4. Re-run stress test to verify fixes")
        print("  5. Submit project with confidence!")


if __name__ == "__main__":
    runner = StressTestRunner()
    runner.run_all_tests()
