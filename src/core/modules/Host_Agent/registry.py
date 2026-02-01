# registry.py
"""
Agent Registry Module

Manages agent configurations, lazy loading, and caching for the Host Agent system.
Supports dependency injection of models and provides thread-safe agent instantiation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration for a single TOEIC Agent
    
    Attributes:
        name: Human-readable agent name (e.g., "TOEIC Part 1 - Image Description")
        agent_class: The agent class to instantiate (must have __init__(model=...))
        model_key: Key to lookup model in models dict (e.g., "gemini", "mistral")
        description: Brief description of agent's purpose
        enabled: Whether this agent is active
        extra_kwargs: Additional kwargs to pass to agent constructor
    """
    name: str
    agent_class: Type[Any]
    model_key: str
    description: str = ""
    enabled: bool = True
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.name:
            raise ValueError("AgentConfig requires a name")
        if not self.agent_class:
            raise ValueError("AgentConfig requires an agent_class")
        if not self.model_key:
            raise ValueError("AgentConfig requires a model_key")


class AgentRegistry:
    """
    Central registry for managing TOEIC agents
    
    Features:
    - Lazy loading: Agents are only instantiated when first requested
    - Caching: Instantiated agents are cached for reuse
    - Model injection: Models are injected at runtime
    - Thread-safe: Uses locks for concurrent access
    
    Usage:
        registry = AgentRegistry()
        registry.register("part1", AgentConfig(...))
        registry.set_models({"gemini": model1, "mistral": model2})
        
        agent = registry.get_agent("part1")  # Lazy loads and caches
    """
    
    def __init__(self):
        self._configs: Dict[str, AgentConfig] = {}
        self._agents: Dict[str, Any] = {}  # Cached agent instances
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        logger.info("[Registry] Initialized")
    
    # ==================== REGISTRATION ====================
    
    def register(self, key: str, config: AgentConfig) -> "AgentRegistry":
        """
        Register an agent configuration
        
        Args:
            key: Unique identifier (e.g., "part1", "part5")
            config: AgentConfig instance
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If key already registered
        """
        if key in self._configs:
            logger.warning(f"[Registry] Overwriting existing config for '{key}'")
        
        self._configs[key] = config
        logger.info(f"[Registry] Registered: {key} -> {config.name}")
        
        return self
    
    def register_many(self, configs: Dict[str, AgentConfig]) -> "AgentRegistry":
        """
        Register multiple agents at once
        
        Args:
            configs: Dict mapping keys to AgentConfig instances
        
        Returns:
            Self for method chaining
        """
        for key, config in configs.items():
            self.register(key, config)
        return self
    
    def unregister(self, key: str) -> bool:
        """
        Remove an agent from registry
        
        Args:
            key: Agent key to remove
        
        Returns:
            True if removed, False if not found
        """
        if key in self._configs:
            del self._configs[key]
            # Also remove cached instance
            if key in self._agents:
                del self._agents[key]
            logger.info(f"[Registry] Unregistered: {key}")
            return True
        return False
    
    # ==================== MODEL MANAGEMENT ====================
    
    def set_models(self, models: Dict[str, Any]) -> "AgentRegistry":
        """
        Set the model instances to be used by agents
        
        Args:
            models: Dict mapping model_key to model instance
                    e.g., {"gemini": ChatGoogleGenerativeAI(...),
                           "mistral": ChatMistralAI(...)}
        
        Returns:
            Self for method chaining
        """
        self._models = models
        logger.info(f"[Registry] Models set: {list(models.keys())}")
        
        # Clear cached agents since models changed
        self._agents.clear()
        logger.info("[Registry] Agent cache cleared (models updated)")
        
        return self
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """
        Get a model instance by key
        
        Args:
            model_key: Key to lookup (e.g., "gemini")
        
        Returns:
            Model instance or None
        """
        return self._models.get(model_key)
    
    # ==================== AGENT ACCESS ====================
    
    def get_config(self, key: str) -> Optional[AgentConfig]:
        """
        Get agent configuration by key
        
        Args:
            key: Agent key (e.g., "part1")
        
        Returns:
            AgentConfig or None if not found
        """
        return self._configs.get(key)
    
    def get_agent(self, key: str) -> Optional[Any]:
        """
        Get or create agent instance (lazy loading with caching)
        
        This method is thread-safe and will only instantiate
        each agent once, reusing cached instances.
        
        Args:
            key: Agent key (e.g., "part1")
        
        Returns:
            Agent instance or None if creation fails
        """
        # Check cache first (without lock for read)
        if key in self._agents:
            logger.debug(f"[Registry] Cache hit: {key}")
            return self._agents[key]
        
        # Need to create - acquire lock
        with self._lock:
            # Double-check after acquiring lock
            if key in self._agents:
                return self._agents[key]
            
            # Get config
            config = self._configs.get(key)
            if not config:
                logger.error(f"[Registry] No config found for '{key}'")
                return None
            
            # Check if enabled
            if not config.enabled:
                logger.warning(f"[Registry] Agent '{key}' is disabled")
                return None
            
            # Get model
            model = self._models.get(config.model_key)
            if not model:
                logger.error(
                    f"[Registry] Model '{config.model_key}' not found for agent '{key}'. "
                    f"Available models: {list(self._models.keys())}"
                )
                return None
            
            # Instantiate agent
            try:
                logger.info(f"[Registry] Instantiating: {key} ({config.name})")
                
                agent = config.agent_class(
                    model=model,
                    **config.extra_kwargs
                )
                
                # Cache the instance
                self._agents[key] = agent
                logger.info(f"[Registry] Successfully created: {key}")
                
                return agent
                
            except Exception as e:
                logger.error(f"[Registry] Failed to instantiate '{key}': {e}")
                return None
    
    def get_agent_or_raise(self, key: str) -> Any:
        """
        Get agent instance, raising exception if not available
        
        Args:
            key: Agent key
        
        Returns:
            Agent instance
        
        Raises:
            KeyError: If agent not found or disabled
            RuntimeError: If instantiation fails
        """
        agent = self.get_agent(key)
        if agent is None:
            config = self._configs.get(key)
            if not config:
                raise KeyError(f"Agent '{key}' not registered")
            if not config.enabled:
                raise KeyError(f"Agent '{key}' is disabled")
            raise RuntimeError(f"Failed to instantiate agent '{key}'")
        return agent
    
    # ==================== INTROSPECTION ====================
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered agents with their status
        
        Returns:
            Dict with agent info including:
            - name, description, enabled, cached, model_key
        """
        result = {}
        for key, config in self._configs.items():
            result[key] = {
                "name": config.name,
                "description": config.description,
                "enabled": config.enabled,
                "model_key": config.model_key,
                "cached": key in self._agents,
                "model_available": config.model_key in self._models
            }
        return result
    
    def list_enabled(self) -> list[str]:
        """Get list of enabled agent keys"""
        return [k for k, c in self._configs.items() if c.enabled]
    
    def list_disabled(self) -> list[str]:
        """Get list of disabled agent keys"""
        return [k for k, c in self._configs.items() if not c.enabled]
    
    @property
    def agent_count(self) -> int:
        """Total number of registered agents"""
        return len(self._configs)
    
    @property
    def cached_count(self) -> int:
        """Number of currently cached agent instances"""
        return len(self._agents)
    
    # ==================== UTILITY ====================
    
    def clear_cache(self) -> None:
        """Clear all cached agent instances"""
        with self._lock:
            self._agents.clear()
        logger.info("[Registry] Cache cleared")
    
    def enable_agent(self, key: str) -> bool:
        """Enable a disabled agent"""
        config = self._configs.get(key)
        if config:
            config.enabled = True
            logger.info(f"[Registry] Enabled: {key}")
            return True
        return False
    
    def disable_agent(self, key: str) -> bool:
        """Disable an agent"""
        config = self._configs.get(key)
        if config:
            config.enabled = False
            # Remove from cache
            if key in self._agents:
                del self._agents[key]
            logger.info(f"[Registry] Disabled: {key}")
            return True
        return False
    
    def __repr__(self) -> str:
        return (
            f"AgentRegistry("
            f"agents={self.agent_count}, "
            f"cached={self.cached_count}, "
            f"models={list(self._models.keys())})"
        )
    
    def __contains__(self, key: str) -> bool:
        """Check if agent is registered: 'part1' in registry"""
        return key in self._configs
