import boto3
import json
from typing import Dict, Any, Optional
import logging


class RetrainingTrigger:

    def __init__(self, step_function_arn: str, region_name: str = "us-east-1"):
        self.step_function_client = boto3.client(
            "stepfunctions", region_name=region_name
        )
        self.step_function_arn = step_function_arn
        self.logger = logging.getLogger(__name__)

    def trigger_retraining(
        self, input_data: Dict[str, Any], execution_name: Optional[str] = None
    ) -> str:
        try:
            input_json = json.dumps(input_data)
            execution_params = {
                "stateMachineArn": self.step_function_arn,
                "input": input_json,
            }
            if execution_name:
                execution_params["name"] = execution_name
            response = self.step_function_client.start_execution(**execution_params)
            execution_arn = response["executionArn"]
            self.logger.info(f"Retraining workflow triggered: {execution_arn}")
            return execution_arn
        except Exception as e:
            self.logger.error(f"Failed to trigger retraining: {str(e)}")
            raise

    def check_execution_status(self, execution_arn: str) -> str:
        try:
            response = self.step_function_client.describe_execution(
                executionArn=execution_arn
            )
            status = response["status"]
            self.logger.info(f"Execution {execution_arn} status: {status}")
            return status
        except Exception as e:
            self.logger.error(f"Failed to check execution status: {str(e)}")
            raise

    def get_execution_details(self, execution_arn: str) -> Dict[str, Any]:
        try:
            response = self.step_function_client.describe_execution(
                executionArn=execution_arn
            )
            details = {
                "status": response["status"],
                "start_date": response["startDate"].isoformat(),
                "state_machine_arn": response["stateMachineArn"],
                "execution_arn": response["executionArn"],
            }
            if "stopDate" in response:
                details["stop_date"] = response["stopDate"].isoformat()
            if response["status"] == "SUCCEEDED" and "output" in response:
                details["output"] = json.loads(response["output"])
            return details
        except Exception as e:
            self.logger.error(f"Failed to get execution details: {str(e)}")
            raise

    def wait_for_completion(
        self, execution_arn: str, max_attempts: int = 60, poll_interval: int = 10
    ) -> Dict[str, Any]:
        import time

        terminal_states = ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]
        for attempt in range(max_attempts):
            status = self.check_execution_status(execution_arn)
            if status in terminal_states:
                self.logger.info(f"Execution completed with status: {status}")
                return self.get_execution_details(execution_arn)
            self.logger.debug(
                f"Waiting for execution completion... (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Execution {execution_arn} did not complete within {max_attempts * poll_interval} seconds"
        )

    def stop_execution(
        self,
        execution_arn: str,
        error: str = "Manual stop",
        cause: str = "Stopped by user",
    ) -> Dict[str, Any]:
        try:
            response = self.step_function_client.stop_execution(
                executionArn=execution_arn, error=error, cause=cause
            )
            self.logger.warning(f"Execution stopped: {execution_arn}")
            return response
        except Exception as e:
            self.logger.error(f"Failed to stop execution: {str(e)}")
            raise
