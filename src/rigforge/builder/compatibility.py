"""
兼容性检查模块 - Compatibility Check Module

检查硬件配件之间的兼容性，包括接口、尺寸、功耗等。
Check compatibility between hardware parts, including interfaces, dimensions, power, etc.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import Part, BuildPlan, UserRequirements


def check_compatibility(
    cpu: Optional["Part"],
    motherboard: Optional["Part"],
    memory: Optional["Part"],
    gpu: Optional["Part"],
    psu: Optional["Part"],
    case: Optional["Part"],
    cooler: Optional["Part"],
) -> List[str]:
    """
    检查硬件兼容性 - Check Hardware Compatibility
    
    检查所有硬件配件之间的兼容性约束。
    Check compatibility constraints between all hardware parts.
    
    检查项 Check Items:
    1. CPU 与主板插槽兼容
    2. 内存与主板 DDR 类型兼容
    3. 显卡与机箱尺寸兼容
    4. 散热器与机箱高度兼容
    5. 电源功率是否足够
    6. 主板与机箱板型兼容
    
    参数 Parameters:
        cpu: CPU 配件
             CPU part
        motherboard: 主板配件
                     Motherboard part
        memory: 内存配件
                 Memory part
        gpu: 显卡配件
             GPU part
        psu: 电源配件
            PSU part
        case: 机箱配件
              Case part
        cooler: 散热器配件
                Cooler part
    
    返回 Returns:
        兼容性问题列表，空列表表示无问题
        List of compatibility issues, empty list means no issues
    """
    issues: List[str] = []
    
    # 检查必需配件是否存在 - Check if required parts exist
    if not all([cpu, motherboard, memory, gpu, psu, case, cooler]):
        return ["配置不完整，缺少必要配件"]
    
    # 检查 1: CPU 与主板 Socket 兼容 - Check 1: CPU and motherboard socket compatibility
    if cpu.socket != motherboard.socket:
        issues.append(f"CPU ({cpu.socket}) 与主板 ({motherboard.socket}) 插槽不兼容")
    
    # 检查 2: 内存与主板 DDR 类型兼容 - Check 2: Memory and motherboard DDR type compatibility
    if motherboard.memory_type and memory.memory_type:
        if motherboard.memory_type != memory.memory_type:
            issues.append(
                f"主板支持 {motherboard.memory_type}，但内存是 {memory.memory_type}"
            )
    
    # 检查 3: 显卡与机箱尺寸兼容 - Check 3: GPU and case dimension compatibility
    if gpu.length_mm > case.length_mm:
        issues.append(
            f"显卡长度 {gpu.length_mm}mm 超过机箱限制 {case.length_mm}mm"
        )
    
    # 检查 4: 散热器与机箱高度兼容 - Check 4: Cooler and case height compatibility
    if cooler.height_mm > case.height_mm and cooler.height_mm > 100:
        issues.append(
            f"散热器高度 {cooler.height_mm}mm 超过机箱限制 {case.height_mm}mm"
        )
    
    # 检查 5: 电源功率是否足够 - Check 5: PSU wattage is sufficient
    estimated_power = estimate_system_power(cpu.watt, gpu.watt)
    if psu.watt < estimated_power:
        issues.append(
            f"电源功率 {psu.watt}W 可能不足（建议 {estimated_power}W 以上）"
        )
    
    # 检查 6: 主板与机箱板型兼容 - Check 6: Motherboard and case form factor compatibility
    if motherboard.form_factor and case.form_factor:
        if motherboard.form_factor == "ATX" and case.form_factor == "mATX":
            issues.append("ATX 主板无法装入 mATX 机箱")
    
    return issues


def estimate_system_power(cpu_watt: int, gpu_watt: int) -> int:
    """
    估算系统所需电源功率 - Estimate System Power Requirements
    
    根据 CPU 和显卡的功耗，估算系统所需的总电源功率。
    Estimate total system power requirements based on CPU and GPU power consumption.
    
    计算公式 Calculation Formula:
    总功耗 = (CPU 功耗 + GPU 功耗 + 其他配件约 120W) × 1.35（余量）
    Total Power = (CPU Wattage + GPU Wattage + Other Parts ~120W) × 1.35 (Headroom)
    
    参数 Parameters:
        cpu_watt: CPU 功耗（瓦特）
                  CPU power consumption (watts)
        gpu_watt: 显卡功耗（瓦特）
                  GPU power consumption (watts)
    
    返回 Returns:
        建议的电源功率（瓦特）
        Recommended PSU wattage (watts)
    """
    # 基础功耗 + 其他配件约 120W + 35% 余量
    # Base power + other parts ~120W + 35% headroom
    return int((cpu_watt + gpu_watt + 120) * 1.35)


def validate_build(
    build: "BuildPlan", 
    requirements: "UserRequirements"
) -> List[str]:
    """
    验证配置方案 - Validate Build Plan
    
    验证配置方案的完整性和兼容性。
    Validate completeness and compatibility of build plan.
    
    验证项 Validation Items:
    1. 检查必要配件是否存在
    2. 检查配件之间的兼容性
    3. 检查总价是否超出预算
    
    参数 Parameters:
        build: 配置方案
               Build plan
        requirements: 用户需求
                     User requirements
    
    返回 Returns:
        问题列表，空列表表示验证通过
        List of issues, empty list means validation passed
    """
    issues: List[str] = []
    
    # 验证 1: 检查必要配件是否存在 - Validation 1: Check if required parts exist
    required_parts = {
        "CPU": build.cpu,
        "主板": build.motherboard,
        "内存": build.memory,
        "显卡": build.gpu,
        "电源": build.psu,
        "机箱": build.case,
        "散热器": build.cooler,
    }
    
    for name, part in required_parts.items():
        if part is None:
            issues.append(f"缺少必要配件: {name}")
    
    # 验证 2: 如果有缺失配件，直接返回 - Validation 2: Return early if parts are missing
    if issues:
        return issues
    
    # 验证 3: 检查兼容性 - Validation 3: Check compatibility
    compat_issues = check_compatibility(
        build.cpu, build.motherboard, build.memory,
        build.gpu, build.psu, build.case, build.cooler
    )
    issues.extend(compat_issues)
    
    # 验证 4: 检查预算 - Validation 4: Check budget
    total = build.total_price()
    if total > requirements.budget_max:
        issues.append(
            f"总价 {total} 元超出预算上限 {requirements.budget_max} 元"
        )
    
    return issues
