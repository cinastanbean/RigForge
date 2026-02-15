from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


PartCategory = Literal[
    "cpu",
    "motherboard",
    "memory",
    "storage",
    "gpu",
    "psu",
    "case",
    "cooler",
]


class Part(BaseModel):
    sku: str
    name: str
    category: PartCategory
    brand: str
    price: int
    socket: str = ""
    tdp: int = 0
    score: int = 0
    vram: int = 0
    length_mm: int = 0
    height_mm: int = 0
    watt: int = 0
    memory_type: str = ""
    form_factor: str = ""
    capacity_gb: int = 0
    efficiency: str = ""


class UserRequirements(BaseModel):
    budget_min: int = 5000
    budget_max: int = 10000
    use_case: List[str] = Field(default_factory=list)
    resolution: str = "1080p"
    brand_blacklist: List[str] = Field(default_factory=list)
    prefer_brands: List[str] = Field(default_factory=list)
    need_wifi: bool = False
    need_quiet: bool = False
    need_rgb: bool = False
    need_monitor: Optional[bool] = None
    storage_target_gb: int = 0
    cpu_preference: str = ""
    game_titles: List[str] = Field(default_factory=list)
    monitor_resolution: str = ""
    monitor_refresh_hz: int = 0
    case_size: Literal["mATX", "ATX"] = "ATX"
    priority: Literal["balanced", "budget", "performance"] = "balanced"
    budget_set: bool = False
    use_case_set: bool = False
    resolution_set: bool = False
    monitor_set: bool = False
    storage_set: bool = False
    noise_set: bool = False


class RequirementUpdate(BaseModel):
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    use_case: Optional[List[str]] = None
    resolution: Optional[str] = None
    brand_blacklist: Optional[List[str]] = None
    prefer_brands: Optional[List[str]] = None
    need_wifi: Optional[bool] = None
    need_quiet: Optional[bool] = None
    need_rgb: Optional[bool] = None
    need_monitor: Optional[bool] = None
    storage_target_gb: Optional[int] = None
    cpu_preference: Optional[str] = None
    game_titles: Optional[List[str]] = None
    monitor_resolution: Optional[str] = None
    monitor_refresh_hz: Optional[int] = None
    case_size: Optional[Literal["mATX", "ATX"]] = None
    priority: Optional[Literal["balanced", "budget", "performance"]] = None
    budget_set: Optional[bool] = None
    use_case_set: Optional[bool] = None
    resolution_set: Optional[bool] = None
    monitor_set: Optional[bool] = None
    storage_set: Optional[bool] = None
    noise_set: Optional[bool] = None
    missing_fields: List[str] = Field(default_factory=list)


class RequirementUpdateWithReply(BaseModel):
    requirement_update: RequirementUpdate
    reply: str


class BuildPlan(BaseModel):
    cpu: Optional[Part] = None
    motherboard: Optional[Part] = None
    memory: Optional[Part] = None
    storage: Optional[Part] = None
    gpu: Optional[Part] = None
    psu: Optional[Part] = None
    case: Optional[Part] = None
    cooler: Optional[Part] = None

    def as_dict(self) -> Dict[str, Optional[dict]]:
        return {
            "cpu": self.cpu.model_dump() if self.cpu else None,
            "motherboard": self.motherboard.model_dump() if self.motherboard else None,
            "memory": self.memory.model_dump() if self.memory else None,
            "storage": self.storage.model_dump() if self.storage else None,
            "gpu": self.gpu.model_dump() if self.gpu else None,
            "psu": self.psu.model_dump() if self.psu else None,
            "case": self.case.model_dump() if self.case else None,
            "cooler": self.cooler.model_dump() if self.cooler else None,
        }

    def total_price(self) -> int:
        total = 0
        for key in [
            "cpu",
            "motherboard",
            "memory",
            "storage",
            "gpu",
            "psu",
            "case",
            "cooler",
        ]:
            part = getattr(self, key)
            if part:
                total += part.price
        return total


class ChatResponse(BaseModel):
    reply: str
    requirements: UserRequirements
    build: BuildPlan
    compatibility_issues: List[str] = Field(default_factory=list)
    estimated_power: int = 0
    estimated_performance: str = ""
    enthusiasm_level: Literal["standard", "high"] = "standard"
    response_mode: Literal["llm", "fallback"] = "fallback"
    fallback_reason: Optional[str] = None
    session_model_provider: Literal["zhipu", "openrouter", "rules"] = "rules"
    turn_model_provider: Literal["zhipu", "openrouter", "rules"] = "rules"
    model_name: str = ""
    model_status_detail: str = ""
    build_data_source: str = "csv(jd+newegg)"
    build_data_version: str = "v0"
    build_data_mode: Literal["jd_newegg", "jd", "newegg"] = "jd_newegg"
    template_history: Dict[str, List[int]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    enthusiasm_level: Optional[Literal["standard", "high"]] = None
    build_data_mode: Optional[Literal["jd_newegg", "jd", "newegg"]] = None

