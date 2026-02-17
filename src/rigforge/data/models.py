"""
数据模型定义（重构版）- Data Models Definition (Refactored Version)

定义系统中使用的所有数据模型。
Define all data models used in the system.

关键改进 Key Improvements:
1. 使用 Optional + None 替代 *_set 标志位
   Use Optional + None instead of *_set flags
2. 按类别分组字段，提高可读性
   Group fields by category for better readability
3. 添加字段描述和验证
   Add field descriptions and validation
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Part(BaseModel):
    """
    配件实体 - Part Entity
    
    表示单个硬件配件的所有信息。
    Represents all information of a single hardware part.
    """
    # 标识 - Identification
    sku: str = Field(description="唯一标识")
    name: str = Field(description="产品名称")
    category: str = Field(description="类别")
    brand: str = Field(description="品牌")
    
    # 价格 - Price
    price: int = Field(description="价格（人民币）", ge=0)
    
    # 兼容性参数 - Compatibility Parameters
    socket: str = Field(default="", description="CPU/主板插槽")
    tdp: int = Field(default=0, ge=0, description="热设计功耗")
    memory_type: str = Field(default="", description="内存类型 DDR4/DDR5")
    form_factor: str = Field(default="", description="板型 ATX/mATX")
    
    # 性能参数 - Performance Parameters
    score: int = Field(default=0, ge=0, description="性能评分")
    vram: int = Field(default=0, ge=0, description="显存容量(GB)")
    watt: int = Field(default=0, ge=0, description="功耗(W)")
    
    # 尺寸 - Dimensions
    length_mm: int = Field(default=0, ge=0, description="长度(mm)")
    height_mm: int = Field(default=0, ge=0, description="高度(mm)")
    
    # 存储 - Storage
    capacity_gb: int = Field(default=0, ge=0, description="容量(GB)")
    efficiency: str = Field(default="", description="电源效率")


class UserRequirements(BaseModel):
    """
    用户需求（重构版）- User Requirements (Refactored Version)
    
    表示用户的装机需求。
    Represents user's build requirements.
    
    None 表示未设置，有值表示已设置
    None means not set, having a value means set
    不再需要 *_set 标志位
    No longer need *_set flags
    """
    # === 核心需求 - Core Requirements ===
    budget_min: Optional[int] = Field(default=None, description="预算下限")
    budget_max: Optional[int] = Field(default=None, description="预算上限")
    use_case: Optional[List[str]] = Field(default=None, description="用途")
    resolution: Optional[str] = Field(default=None, description="分辨率")
    priority: Optional[Literal["budget", "balanced", "performance"]] = Field(
        default=None, description="优先级"
    )
    
    # === 配件偏好 - Component Preferences ===
    cpu_preference: Optional[str] = Field(default=None, description="CPU偏好 Intel/AMD")
    cpu_model: Optional[str] = Field(default=None, description="指定CPU型号")
    gpu_preference: Optional[str] = Field(default=None, description="显卡偏好")
    gpu_model: Optional[str] = Field(default=None, description="指定显卡型号")
    memory_gb: Optional[int] = Field(default=None, description="内存容量(GB)")
    memory_type: Optional[str] = Field(default=None, description="内存类型")
    storage_target_gb: Optional[int] = Field(default=None, description="存储容量(GB)")
    
    # === 其他偏好 - Other Preferences ===
    prefer_brands: Optional[List[str]] = Field(default=None, description="品牌偏好")
    brand_blacklist: Optional[List[str]] = Field(default=None, description="禁用品牌")
    need_wifi: Optional[bool] = Field(default=None, description="需要WiFi")
    need_quiet: Optional[bool] = Field(default=None, description="需要静音")
    need_rgb: Optional[bool] = Field(default=None, description="需要RGB")
    case_size: Optional[Literal["mATX", "ATX"]] = Field(default=None, description="机箱尺寸")
    
    # === 游戏相关 - Game Related ===
    game_titles: Optional[List[str]] = Field(default=None, description="游戏名称")
    
    def is_budget_set(self) -> bool:
        """
        检查预算是否已设置 - Check if budget is set
        
        返回 Returns:
            预算是否已设置
            Whether budget is set
        """
        return self.budget_max is not None
    
    def is_use_case_set(self) -> bool:
        """
        检查用途是否已设置 - Check if use case is set
        
        返回 Returns:
            用途是否已设置
            Whether use case is set
        """
        return self.use_case is not None
    
    def is_resolution_set(self) -> bool:
        """
        检查分辨率是否已设置 - Check if resolution is set
        
        返回 Returns:
            分辨率是否已设置
            Whether resolution is set
        """
        return self.resolution is not None
    
    def get_budget_range(self) -> tuple[int, int]:
        """
        获取预算范围 - Get Budget Range
        
        未设置时返回默认值。
        Return default values when not set.
        
        返回 Returns:
            (预算下限, 预算上限)
            (Budget minimum, Budget maximum)
        """
        return (
            self.budget_min or 5000,
            self.budget_max or 10000
        )


class RequirementUpdate(BaseModel):
    """
    需求增量更新 - Requirement Incremental Update
    
    表示对用户需求的增量更新。
    Represents incremental update to user requirements.
    """
    # 所有字段均可选 - All fields are optional
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    use_case: Optional[List[str]] = None
    resolution: Optional[str] = None
    priority: Optional[Literal["budget", "balanced", "performance"]] = None
    cpu_preference: Optional[str] = None
    cpu_model: Optional[str] = None
    gpu_preference: Optional[str] = None
    gpu_model: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    storage_target_gb: Optional[int] = None
    prefer_brands: Optional[List[str]] = None
    brand_blacklist: Optional[List[str]] = None
    need_wifi: Optional[bool] = None
    need_quiet: Optional[bool] = None
    need_rgb: Optional[bool] = None
    case_size: Optional[Literal["mATX", "ATX"]] = None
    game_titles: Optional[List[str]] = None
    missing_fields: List[str] = Field(default_factory=list)


class BuildPlan(BaseModel):
    """
    装机配置方案 - Build Plan
    
    表示一个完整的硬件配置方案。
    Represents a complete hardware build plan.
    """
    cpu: Optional[Part] = None
    motherboard: Optional[Part] = None
    memory: Optional[Part] = None
    storage: Optional[Part] = None
    gpu: Optional[Part] = None
    psu: Optional[Part] = None
    case: Optional[Part] = None
    cooler: Optional[Part] = None
    
    def total_price(self) -> int:
        """
        计算总价 - Calculate Total Price
        
        返回 Returns:
            配置方案的总价格
            Total price of the build plan
        """
        total = 0
        for attr in ["cpu", "motherboard", "memory", "storage", "gpu", "psu", "case", "cooler"]:
            part = getattr(self, attr)
            if part:
                total += part.price
        return total
    
    def to_dict(self) -> Dict[str, Optional[Part]]:
        """
        转换为字典 - Convert to Dictionary
        
        返回 Returns:
            配置方案的字典表示
            Dictionary representation of the build plan
        """
        return {
            "cpu": self.cpu,
            "motherboard": self.motherboard,
            "memory": self.memory,
            "storage": self.storage,
            "gpu": self.gpu,
            "psu": self.psu,
            "case": self.case,
            "cooler": self.cooler,
        }
