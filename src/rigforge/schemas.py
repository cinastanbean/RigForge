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
"""
配件类别类型 - Part Category Type

定义所有支持的配件类别。
Defines all supported part categories.
"""


class Part(BaseModel):
    """
    配件模型 - Part Model
    
    表示单个硬件配件的详细信息。
    Represents detailed information of a single hardware part.
    
    字段说明 Field Descriptions:
    - sku: 配件唯一标识符
    - name: 配件名称
    - category: 配件类别
    - brand: 品牌
    - price: 价格（元）
    - socket: 插槽类型（CPU/主板）
    - tdp: 热设计功耗（瓦特）
    - score: 性能评分（0-100）
    - vram: 显存容量（GB，仅显卡）
    - length_mm: 长度（毫米，显卡）
    - height_mm: 高度（毫米，散热器/机箱）
    - watt: 功耗（瓦特）
    - memory_type: 内存类型（DDR4/DDR5）
    - form_factor: 规格类型（ATX/mATX）
    - capacity_gb: 容量（GB，内存/存储）
    - efficiency: 效率等级（电源）
    """
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
    """
    用户需求模型 - User Requirements Model
    
    表示用户的装机需求和偏好。
    Represents user's PC build requirements and preferences.
    
    字段分组 Field Groups:
    1. 核心需求: 预算、用途、分辨率、优先级
    2. 配件偏好: 具体型号/规格偏好
    3. 其他偏好: 品牌、功能需求等
    4. 状态标记: 标记哪些字段已被设置
    """
    
    # === 核心需求 Core Requirements ===
    budget_min: int = 5000
    """
    最小预算 - Minimum Budget
    
    用户可接受的最低预算（元）。
    Minimum acceptable budget (CNY).
    """
    budget_max: int = 10000
    """
    最大预算 - Maximum Budget
    
    用户可接受的最高预算（元）。
    Maximum acceptable budget (CNY).
    """
    use_case: List[str] = Field(default_factory=list)
    """
    用途列表 - Use Case List
    
    电脑的主要用途，如 ["gaming"], ["video_editing", "ai"]。
    Primary use cases of the PC, such as ["gaming"], ["video_editing", "ai"].
    """
    resolution: str = "1080p"
    """
    分辨率 - Resolution
    
    显示器分辨率，如 "1080p", "1440p", "4k"。
    Monitor resolution, such as "1080p", "1440p", "4k".
    """
    priority: Literal["balanced", "budget", "performance"] = "balanced"
    """
    优先级 - Priority
    
    配置优先级：balanced（平衡）、budget（性价比）、performance（性能）。
    Configuration priority: balanced, budget, or performance.
    """
    
    # === 配件偏好（具体型号/规格）Part Preferences (Specific Models/Specs) ===
    # CPU
    cpu_preference: str = ""
    """
    CPU 偏好 - CPU Preference
    
    品牌偏好（Intel/AMD）或具体型号。
    Brand preference (Intel/AMD) or specific model.
    """
    cpu_model: str = ""
    """
    CPU 型号 - CPU Model
    
    用户指定的具体 CPU 型号。
    Specific CPU model specified by user.
    """
    
    # 显卡 GPU
    gpu_preference: str = ""
    """
    显卡偏好 - GPU Preference
    
    品牌偏好（NVIDIA/AMD）或具体型号。
    Brand preference (NVIDIA/AMD) or specific model.
    """
    gpu_model: str = ""
    """
    显卡型号 - GPU Model
    
    用户指定的具体显卡型号。
    Specific GPU model specified by user.
    """
    
    # 内存 Memory
    memory_gb: int = 0
    """
    内存容量 - Memory Capacity
    
    内存容量需求（GB），如 16, 32, 64。
    Memory capacity requirement (GB), such as 16, 32, 64.
    """
    memory_type: str = ""
    """
    内存类型 - Memory Type
    
    内存类型偏好，如 DDR4/DDR5。
    Memory type preference, such as DDR4/DDR5.
    """
    
    # 存储 Storage
    storage_target_gb: int = 0
    """
    存储容量 - Storage Capacity
    
    存储容量需求（GB）。
    Storage capacity requirement (GB).
    """
    storage_type: str = ""
    """
    存储类型 - Storage Type
    
    存储类型偏好，如 NVMe/SATA SSD。
    Storage type preference, such as NVMe/SATA SSD.
    """
    
    # 主板 Motherboard
    motherboard_preference: str = ""
    """
    主板偏好 - Motherboard Preference
    
    品牌或具体型号偏好。
    Brand or specific model preference.
    """
    
    # 机箱 Case
    case_size: Literal["mATX", "ATX"] = "ATX"
    """
    机箱尺寸 - Case Size
    
    机箱尺寸规格：mATX 或 ATX。
    Case size specification: mATX or ATX.
    """
    case_preference: str = ""
    """
    机箱偏好 - Case Preference
    
    机箱风格偏好（静音/RGB/紧凑等）。
    Case style preference (quiet/RGB/compact, etc.).
    """
    
    # 电源 PSU
    psu_wattage: int = 0
    """
    电源功率 - PSU Wattage
    
    电源功率需求（瓦特）。
    PSU wattage requirement (watts).
    """
    psu_efficiency: str = ""
    """
    电源效率 - PSU Efficiency
    
    电源效率等级，如金牌/白金牌。
    PSU efficiency rating, such as Gold/Platinum.
    """
    
    # === 其他偏好 Other Preferences ===
    brand_blacklist: List[str] = Field(default_factory=list)
    """
    品牌黑名单 - Brand Blacklist
    
    不希望使用的品牌列表。
    List of brands to avoid.
    """
    prefer_brands: List[str] = Field(default_factory=list)
    """
    品牌偏好列表 - Preferred Brands List
    
    优先选择的品牌列表。
    List of preferred brands.
    """
    need_wifi: bool = False
    """
    是否需要 WiFi - Need WiFi
    
    是否需要内置 WiFi 功能。
    Whether built-in WiFi is needed.
    """
    need_quiet: bool = False
    """
    是否需要静音 - Need Quiet
    
    是否需要静音配置。
    Whether quiet configuration is needed.
    """
    need_rgb: bool = False
    """
    是否需要 RGB - Need RGB
    
    是否需要 RGB 灯效。
    Whether RGB lighting is needed.
    """
    need_monitor: Optional[bool] = None
    """
    是否需要显示器 - Need Monitor
    
    是否需要显示器（已废弃，不再使用）。
    Whether monitor is needed (deprecated, no longer used).
    """
    game_titles: List[str] = Field(default_factory=list)
    """
    游戏列表 - Game Titles
    
    主要运行的游戏列表，用于显卡推荐。
    List of main games to play, used for GPU recommendation.
    """
    monitor_resolution: str = ""
    """
    显示器分辨率 - Monitor Resolution
    
    显示器分辨率（已废弃，不再使用）。
    Monitor resolution (deprecated, no longer used).
    """
    monitor_refresh_hz: int = 0
    """
    显示器刷新率 - Monitor Refresh Rate
    
    显示器刷新率（已废弃，不再使用）。
    Monitor refresh rate (deprecated, no longer used).
    """
    
    # === 状态标记 Status Flags ===
    budget_set: bool = False
    """
    预算已设置 - Budget Set
    
    标记预算是否已被用户设置。
    Flag indicating if budget has been set by user.
    """
    use_case_set: bool = False
    """
    用途已设置 - Use Case Set
    
    标记用途是否已被用户设置。
    Flag indicating if use case has been set by user.
    """
    resolution_set: bool = False
    """
    分辨率已设置 - Resolution Set
    
    标记分辨率是否已被用户设置。
    Flag indicating if resolution has been set by user.
    """
    monitor_set: bool = False
    """
    显示器已设置 - Monitor Set
    
    标记显示器需求是否已被用户设置（已废弃）。
    Flag indicating if monitor requirement has been set by user (deprecated).
    """
    storage_set: bool = False
    """
    存储已设置 - Storage Set
    
    标记存储需求是否已被用户设置。
    Flag indicating if storage requirement has been set by user.
    """
    noise_set: bool = False
    """
    静音需求已设置 - Noise Set
    
    标记静音需求是否已被用户设置。
    Flag indicating if quiet requirement has been set by user.
    """
    cpu_set: bool = False
    """
    CPU 偏好已设置 - CPU Set
    
    标记 CPU 偏好是否已被用户设置。
    Flag indicating if CPU preference has been set by user.
    """
    gpu_set: bool = False
    """
    显卡偏好已设置 - GPU Set
    
    标记显卡偏好是否已被用户设置。
    Flag indicating if GPU preference has been set by user.
    """
    memory_set: bool = False
    """
    内存需求已设置 - Memory Set
    
    标记内存需求是否已被用户设置。
    Flag indicating if memory requirement has been set by user.
    """


class RequirementUpdate(BaseModel):
    """
    需求更新模型 - Requirement Update Model
    
    表示从用户输入中提取的需求更新信息。
    Represents requirement update information extracted from user input.
    
    用途 Usage:
    - 用于 LLM 返回的需求提取结果
    - 所有字段都是可选的，只包含被识别到的字段
    - 用于更新 UserRequirements 对象
    """
    
    # === 核心需求 Core Requirements ===
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    use_case: Optional[List[str]] = None
    resolution: Optional[str] = None
    priority: Optional[Literal["balanced", "budget", "performance"]] = None
    
    # === 配件偏好 Part Preferences ===
    cpu_preference: Optional[str] = None
    cpu_model: Optional[str] = None
    gpu_preference: Optional[str] = None
    gpu_model: Optional[str] = None
    memory_gb: Optional[int] = None
    memory_type: Optional[str] = None
    storage_target_gb: Optional[int] = None
    storage_type: Optional[str] = None
    motherboard_preference: Optional[str] = None
    case_size: Optional[Literal["mATX", "ATX"]] = None
    case_preference: Optional[str] = None
    psu_wattage: Optional[int] = None
    psu_efficiency: Optional[str] = None
    
    # === 其他偏好 Other Preferences ===
    brand_blacklist: Optional[List[str]] = None
    prefer_brands: Optional[List[str]] = None
    need_wifi: Optional[bool] = None
    need_quiet: Optional[bool] = None
    need_rgb: Optional[bool] = None
    need_monitor: Optional[bool] = None
    game_titles: Optional[List[str]] = None
    monitor_resolution: Optional[str] = None
    monitor_refresh_hz: Optional[int] = None
    
    # === 状态标记 Status Flags ===
    budget_set: Optional[bool] = None
    use_case_set: Optional[bool] = None
    resolution_set: Optional[bool] = None
    monitor_set: Optional[bool] = None
    storage_set: Optional[bool] = None
    noise_set: Optional[bool] = None
    cpu_set: Optional[bool] = None
    gpu_set: Optional[bool] = None
    memory_set: Optional[bool] = None
    
    missing_fields: List[str] = Field(default_factory=list)
    """
    缺失字段列表 - Missing Fields List
    
    还需要收集的字段列表。
    List of fields that still need to be collected.
    """


class RequirementUpdateWithReply(BaseModel):
    """
    带回复的需求更新模型 - Requirement Update with Reply Model
    
    包含需求更新、LLM 生成的回复和是否继续对话的标志。
    Contains requirement update, LLM-generated reply, and flag for continuing conversation.
    
    用途 Usage:
    - 用于对话模式的需求提取和回复生成
    - LLM 一次性返回需求更新和对话回复
    """
    requirement_update: RequirementUpdate
    """
    需求更新 - Requirement Update
    
    从用户输入中提取的需求更新信息。
    Requirement update information extracted from user input.
    """
    reply: str
    """
    回复内容 - Reply Content
    
    LLM 生成的回复内容，可能包含问题或确认信息。
    LLM-generated reply content, may contain questions or confirmation.
    """
    should_continue: bool = True
    """
    是否继续对话 - Should Continue
    
    标记是否需要继续收集需求信息。
    Flag indicating whether to continue collecting requirement information.
    """


class BuildPlan(BaseModel):
    """
    配置方案模型 - Build Plan Model
    
    表示完整的硬件配置方案，包含所有选定的配件。
    Represents complete hardware configuration plan, including all selected parts.
    
    字段 Fields:
    - cpu: CPU 配件
    - motherboard: 主板配件
    - memory: 内存配件
    - storage: 存储配件
    - gpu: 显卡配件
    - psu: 电源配件
    - case: 机箱配件
    - cooler: 散热器配件
    """
    cpu: Optional[Part] = None
    motherboard: Optional[Part] = None
    memory: Optional[Part] = None
    storage: Optional[Part] = None
    gpu: Optional[Part] = None
    psu: Optional[Part] = None
    case: Optional[Part] = None
    cooler: Optional[Part] = None

    def as_dict(self) -> Dict[str, Optional[dict]]:
        """
        转换为字典 - Convert to Dictionary
        
        将配置方案转换为字典格式，便于序列化和传输。
        Convert build plan to dictionary format for serialization and transmission.
        
        返回 Returns:
            配件字典，键为配件类别，值为配件字典或 None
            Dictionary of parts, keys are part categories, values are part dictionaries or None
        """
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
        """
        计算总价 - Calculate Total Price
        
        计算配置方案的总价格。
        Calculate total price of the build plan.
        
        返回 Returns:
            总价格（元）
            Total price (CNY)
        """
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
    """
    聊天响应模型 - Chat Response Model
    
    表示聊天接口的完整响应，包含回复、需求、配置方案和元数据。
    Represents complete response from chat interface, including reply, requirements, build plan, and metadata.
    
    字段分组 Field Groups:
    1. 主要内容: 回复、需求、配置方案
    2. 配置信息: 兼容性问题、功耗、性能
    3. 交互控制: 热情度、响应模式、回退原因
    4. 模型信息: 会话模型、轮次模型、模型名称、状态详情
    5. 数据源信息: 数据源、版本、模式
    6. 模板历史: 模板使用历史
    """
    reply: str
    """
    回复内容 - Reply Content
    
    返回给用户的回复文本。
    Reply text returned to user.
    """
    requirements: UserRequirements
    """
    用户需求 - User Requirements
    
    当前的用户需求状态。
    Current user requirements state.
    """
    build: BuildPlan
    """
    配置方案 - Build Plan
    
    推荐的硬件配置方案。
    Recommended hardware configuration plan.
    """
    compatibility_issues: List[str] = Field(default_factory=list)
    """
    兼容性问题 - Compatibility Issues
    
    配置方案的兼容性问题列表。
    List of compatibility issues in the build plan.
    """
    estimated_power: int = 0
    """
    估算功耗 - Estimated Power
    
    配置方案的估算功耗（瓦特）。
    Estimated power draw of the build plan (watts).
    """
    estimated_performance: str = ""
    """
    估算性能 - Estimated Performance
    
    配置方案的估算性能描述。
    Estimated performance description of the build plan.
    """
    enthusiasm_level: Literal["standard", "high"] = "standard"
    """
    热情度 - Enthusiasm Level
    
    回复的热情度：standard（标准）、high（高）。
    Reply enthusiasm level: standard or high.
    """
    response_mode: Literal["llm", "fallback"] = "fallback"
    """
    响应模式 - Response Mode
    
    响应模式：llm（LLM 生成）、fallback（规则回退）。
    Response mode: llm (LLM generated) or fallback (rule-based).
    """
    fallback_reason: Optional[str] = None
    """
    回退原因 - Fallback Reason
    
    使用规则回退的原因。
    Reason for using rule-based fallback.
    """
    session_model_provider: Literal["zhipu", "openrouter", "rules"] = "rules"
    """
    会话模型提供商 - Session Model Provider
    
    会话级别的模型提供商：zhipu、openrouter 或 rules。
    Session-level model provider: zhipu, openrouter, or rules.
    """
    turn_model_provider: Literal["zhipu", "openrouter", "rules"] = "rules"
    """
    轮次模型提供商 - Turn Model Provider
    
    当前轮次的模型提供商：zhipu、openrouter 或 rules。
    Current turn model provider: zhipu, openrouter, or rules.
    """
    model_name: str = ""
    """
    模型名称 - Model Name
    
    使用的模型名称。
    Name of the model used.
    """
    model_status_detail: str = ""
    """
    模型状态详情 - Model Status Detail
    
    模型状态的详细信息。
    Detailed information about model status.
    """
    build_data_source: str = "csv(jd+newegg)"
    """
    配置数据源 - Build Data Source
    
    配件数据源，如 "csv(jd+newegg)"。
    Parts data source, such as "csv(jd+newegg)".
    """
    build_data_version: str = "v0"
    """
    配置数据版本 - Build Data Version
    
    配件数据版本。
    Parts data version.
    """
    build_data_mode: Literal["jd_newegg", "jd", "newegg"] = "jd_newegg"
    """
    配置数据模式 - Build Data Mode
    
    配件数据模式：jd_newegg（京东+新蛋）、jd（仅京东）、newegg（仅新蛋）。
    Parts data mode: jd_newegg, jd, or newegg.
    """
    template_history: Dict[str, List[int]] = Field(default_factory=dict)
    """
    模板历史 - Template History
    
    模板使用历史记录。
    Template usage history.
    """


class ChatRequest(BaseModel):
    """
    聊天请求模型 - Chat Request Model
    
    表示聊天接口的请求参数。
    Represents request parameters for chat interface.
    
    字段说明 Field Descriptions:
    - session_id: 会话 ID，用于标识会话
    - message: 用户消息
    - interaction_mode: 交互模式（对话/组件选择）
    - enthusiasm_level: 热情度（标准/高）
    - build_data_mode: 配件数据模式
    """
    session_id: str
    """
    会话 ID - Session ID
    
    用于标识会话的唯一 ID。
    Unique ID for identifying the session.
    """
    message: str
    """
    用户消息 - User Message
    
    用户输入的消息文本。
    Message text input by user.
    """
    interaction_mode: Optional[Literal["chat", "component"]] = None
    """
    交互模式 - Interaction Mode
    
    交互模式：chat（对话模式）、component（组件选择模式）。
    Interaction mode: chat or component.
    """
    enthusiasm_level: Optional[Literal["standard", "high"]] = None
    """
    热情度 - Enthusiasm Level
    
    回复的热情度：standard（标准）、high（高）。
    Reply enthusiasm level: standard or high.
    """
    build_data_mode: Optional[Literal["jd_newegg", "jd", "newegg"]] = None
    """
    配件数据模式 - Build Data Mode
    
    配件数据模式：jd_newegg（京东+新蛋）、jd（仅京东）、newegg（仅新蛋）。
    Parts data mode: jd_newegg, jd, or newegg.
    """

