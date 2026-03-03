from typing import Any, Dict, List, Optional
import time
from azure.ai.projects import AIProjectClient
from dataclasses import dataclass
import json

@dataclass
class AgentResponse:
    answer: str
    confidence: float
    reasoning: str
    latency: float

class AzureAgent:
    """Base agent interface. Subclass to implement `call` using a model or SDK."""

    def __init__(self, client: AIProjectClient, model: str, sys_prompt: str, tools: Optional[List[str]] = None, **kwargs):
        self.client = client
        self.model = model
        self.sys_prompt = sys_prompt
        self.tools = tools or []
        self.agent = self.client.agents.create_agent(
                model=self.model,
                name=f"{self.model}" if len(self.tools) == 0 else f"{self.model} with RAG",
                instructions=self.sys_prompt
        )


    def run(self, message: str) -> AgentResponse:
        agent = self.agent
        thread = self.client.agents.threads.create()
        self.client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=message
        )
        
        # Run agent
        run = self.client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id
        )
        
        # Check run status
        if run.status == "failed":
            error_msg = f"Agent run failed. Status: {run.status}"
            if hasattr(run, 'last_error') and run.last_error:
                error_msg += f", Error: {run.last_error}"
            raise ValueError(error_msg)
        
        # Get response - filter for assistant messages only
        messages = self.client.agents.messages.list(thread_id=thread.id)
        messages_list = list(messages)
        assistant_messages = [msg for msg in messages_list if msg.role == "assistant"]
        
        if not assistant_messages:
            error_msg = f"No assistant response found. Run status: {run.status}"
            if hasattr(run, 'last_error') and run.last_error:
                error_msg += f", Error: {run.last_error}"
            raise ValueError(error_msg)
        
        agent_response = assistant_messages[-1].content[0].text.value
        
        try:
            # Parse JSON response
            response_text = agent_response.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response_text)
            answer = parsed.get("answer")
            reasoning = parsed.get("reasoning")
            confidence = parsed.get("confidence_score")
            latency = round((run.completed_at - run.created_at).total_seconds(), 2)
            
            return AgentResponse(
                answer=answer, 
                reasoning=reasoning, 
                confidence=confidence, 
                latency=latency
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to parse agent's response: {e}")
            return AgentResponse(
                answer="none", 
                reasoning="none", 
                confidence=0, 
                latency=round((run.completed_at - run.created_at).total_seconds(), 2), 
            )
