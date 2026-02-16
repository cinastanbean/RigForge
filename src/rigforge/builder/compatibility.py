"""兼容性检查模块"""

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
    """检查硬件兼容性
    
    Args:
        cpu, motherboard, memory, gpu, psu, case, cooler: 各配件
        
    Returns:
        兼容性问题列表，空列表表示无问题
    """
    issues: List[str] = []
    
    # 检查必需配件是否存在
    if not all([cpu, motherboard, memory, gpu, psu, case, cooler]):
        return ["配置不完整，缺少必要配件"]
    
    # 1. CPU 与主板 Socket 兼容
    if cpu.socket != motherboard.socket:
        issues.append(f"CPU ({cpu.socket}) 与主板 ({motherboard.socket}) 插槽不兼容")
    
    # 2. 内存与主板 DDR 类型兼容
    if motherboard.memory_type and memory.memory_type:
        if motherboard.memory_type != memory.memory_type:
            issues.append(
                f"主板支持 {motherboard.memory_type}，但内存是 {memory.memory_type}"
            )
    
    # 3. 显卡与机箱尺寸兼容
    if gpu.length_mm > case.length_mm:
        issues.append(
            f"显卡长度 {gpu.length_mm}mm 超过机箱限制 {case.length_mm}mm"
        )
    
    # 4. 散热器与机箱高度兼容
    if cooler.height_mm > case.height_mm and cooler.height_mm > 100:
        issues.append(
            f"散热器高度 {cooler.height_mm}mm 超过机箱限制 {case.height_mm}mm"
        )
    
    # 5. 电源功率是否足够
    estimated_power = estimate_system_power(cpu.watt, gpu.watt)
    if psu.watt < estimated_power:
        issues.append(
            f"电源功率 {psu.watt}W 可能不足（建议 {estimated_power}W 以上）"
        )
    
    # 6. 主板与机箱板型兼容
    if motherboard.form_factor and case.form_factor:
        if motherboard.form_factor == "ATX" and case.form_factor == "mATX":
            issues.append("ATX 主板无法装入 mATX 机箱")
    
    return issues


def estimate_system_power(cpu_watt: int, gpu_watt: int) -> int:
    """估算系统所需电源功率
    
    Args:
        cpu_watt: CPU 功耗
        gpu_watt: 显卡功耗
        
    Returns:
        建议的电源功率（W）
    """
    # 基础功耗 + 其他配件约 120W + 35% 余量
    return int((cpu_watt + gpu_watt + 120) * 1.35)


def validate_build(
    build: "BuildPlan", 
    requirements: "UserRequirements"
) -> List[str]:
    """验证配置方案的完整性和兼容性
    
    Args:
        build: 配置方案
        requirements: 用户需求
        
    Returns:
        问题列表，空列表表示验证通过
    """
    issues: List[str] = []
    
    # 1. 检查必要配件是否存在
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
    
    # 2. 如果有缺失配件，直接返回
    if issues:
        return issues
    
    # 3. 检查兼容性
    compat_issues = check_compatibility(
        build.cpu, build.motherboard, build.memory,
        build.gpu, build.psu, build.case, build.cooler
    )
    issues.extend(compat_issues)
    
    # 4. 检查预算
    total = build.total_price()
    if total > requirements.budget_max:
        issues.append(
            f"总价 {total} 元超出预算上限 {requirements.budget_max} 元"
        )
    
    return issues
