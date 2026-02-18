"""
品牌匹配测试 - Brand Matching Tests

测试 CPU 品牌偏好选择功能，确保 Intel/AMD 品牌过滤正确工作。
Test CPU brand preference selection, ensure Intel/AMD brand filtering works correctly.
"""

import pytest
from pathlib import Path

from rigforge.data.repository import SQLitePartsRepository
from rigforge.tools import Toolset, pick_build_from_candidates
from rigforge.schemas import UserRequirements


# 数据库路径 - Database path
DB_PATH = Path(__file__).parent.parent / "data" / "agent_parts.db"


class TestBrandMatching:
    """品牌匹配测试类 - Brand Matching Test Class"""

    @pytest.fixture
    def repo_all(self):
        """所有数据源仓库 - All data sources repository"""
        return SQLitePartsRepository(DB_PATH)

    @pytest.fixture
    def repo_jd(self):
        """京东数据源仓库 - JD data source repository"""
        return SQLitePartsRepository(DB_PATH, source_sites={"jd"})

    @pytest.fixture
    def repo_newegg(self):
        """Newegg 数据源仓库 - Newegg data source repository"""
        return SQLitePartsRepository(DB_PATH, source_sites={"newegg"})

    @pytest.fixture
    def toolset_all(self, repo_all):
        """所有数据源工具集 - All data sources toolset"""
        return Toolset(repo_all)

    def test_database_has_intel_cpu(self, repo_all):
        """测试数据库中有 Intel CPU"""
        cpus = repo_all.by_category("cpu")
        intel_cpus = [p for p in cpus if p.brand.lower() in ("intel", "英特尔")]
        
        print(f"\n[TEST] Total CPUs: {len(cpus)}")
        print(f"[TEST] Intel CPUs: {len(intel_cpus)}")
        
        for p in intel_cpus[:5]:
            print(f"  - {p.brand} {p.name} - {p.price}元")
        
        assert len(intel_cpus) > 0, "数据库中应该有 Intel CPU"

    def test_jd_source_has_intel_cpu(self, repo_jd):
        """测试京东数据源有 Intel CPU"""
        cpus = repo_jd.by_category("cpu")
        intel_cpus = [p for p in cpus if p.brand.lower() in ("intel", "英特尔")]
        
        print(f"\n[TEST] JD CPUs: {len(cpus)}")
        print(f"[TEST] JD Intel CPUs: {len(intel_cpus)}")
        
        assert len(intel_cpus) > 0, "京东数据源应该有 Intel CPU"

    def test_search_parts_with_intel_preference(self, toolset_all):
        """测试搜索 Intel CPU（品牌偏好='Intel'）"""
        tools = toolset_all.register()
        search_parts = tools["search_parts"]
        
        result = search_parts.invoke({
            "category": "cpu",
            "budget_max": 1800,
            "prefer_brands": ["Intel"],
            "exclude_brands": [],
        })
        
        print(f"\n[TEST] Search result count: {len(result)}")
        for i, p in enumerate(result[:5]):
            print(f"  #{i+1}: {p['brand']} {p['name']} - {p['price']}元")
        
        assert len(result) > 0, "搜索 Intel CPU 应该有结果"
        
        # 验证所有结果都是 Intel 品牌
        for p in result:
            brand_lower = p['brand'].lower()
            assert brand_lower in ("intel", "英特尔"), f"结果应该都是 Intel 品牌，但找到: {p['brand']}"

    def test_search_parts_with_chinese_brand_intel(self, toolset_all):
        """测试搜索中文品牌的 Intel CPU"""
        tools = toolset_all.register()
        search_parts = tools["search_parts"]
        
        # 先获取所有 Intel CPU，看看品牌字段
        all_cpus = toolset_all.repo.by_category("cpu")
        intel_cpus = [p for p in all_cpus if "intel" in p.name.lower() or "i3" in p.name.lower() or "i5" in p.name.lower()]
        
        print(f"\n[TEST] Intel CPUs in database:")
        brands_found = set()
        for p in intel_cpus[:10]:
            brands_found.add(p.brand)
            print(f"  - brand='{p.brand}', name='{p.name}'")
        
        # 搜索时应该能匹配中文"英特尔"品牌
        result = search_parts.invoke({
            "category": "cpu",
            "budget_max": 1800,
            "prefer_brands": ["Intel"],
            "exclude_brands": [],
        })
        
        # 检查是否有中文品牌的结果
        chinese_brand_count = sum(1 for p in result if p['brand'] == "英特尔")
        print(f"\n[TEST] Results with Chinese brand '英特尔': {chinese_brand_count}")
        
        # 至少应该有一些结果（不管是中文还是英文品牌）
        assert len(result) > 0, "应该能找到 Intel CPU（包括中文品牌的）"

    def test_pick_build_with_intel_preference(self, toolset_all):
        """测试选择 Intel CPU 进行装机配置"""
        tools = toolset_all.register()
        search_parts = tools["search_parts"]
        
        # 创建用户需求，指定 Intel CPU
        req = UserRequirements(
            budget_min=6000,
            budget_max=9000,
            budget_set=True,
            use_case=["gaming"],
            use_case_set=True,
            resolution="1440p",
            resolution_set=True,
            cpu_preference="Intel",  # 指定 Intel CPU
        )
        
        build = pick_build_from_candidates(req, search_parts)
        
        print(f"\n[TEST] Build result:")
        if build.cpu:
            print(f"  CPU: {build.cpu.brand} {build.cpu.name} - {build.cpu.price}元")
        else:
            print(f"  CPU: None")
        
        # CPU 不应该为空
        assert build.cpu is not None, "应该能选择到 CPU"
        
        # CPU 应该是 Intel 品牌
        cpu_brand_lower = build.cpu.brand.lower()
        assert cpu_brand_lower in ("intel", "英特尔"), f"CPU 应该是 Intel 品牌，但找到: {build.cpu.brand}"


    def test_newegg_mode_no_intel_cpu(self, repo_newegg):
        """测试 Newegg 模式下没有 Intel CPU"""
        cpus = repo_newegg.by_category("cpu")
        intel_cpus = [p for p in cpus if p.brand.lower() in ("intel", "英特尔")]
        
        print(f"\n[TEST] Newegg CPUs: {len(cpus)}")
        print(f"[TEST] Newegg Intel CPUs: {len(intel_cpus)}")
        
        # Newegg 数据源没有 Intel CPU，这是数据问题
        # 用户应该使用 jd_newegg 模式或 jd 模式
        print("[WARNING] Newegg 数据源没有 Intel CPU！用户应使用 jd_newegg 或 jd 模式")

    def test_default_mode_has_intel_cpu(self):
        """测试默认模式（jd）有 Intel CPU"""
        from rigforge.main import repo_jd
        
        cpus = repo_jd.by_category("cpu")
        intel_cpus = [p for p in cpus if p.brand.lower() in ("intel", "英特尔")]
        
        print(f"\n[TEST] Default mode CPUs: {len(cpus)}")
        print(f"[TEST] Default mode Intel CPUs: {len(intel_cpus)}")
        
        assert len(intel_cpus) > 0, "默认模式应该有 Intel CPU"


class TestBrandAliases:
    """品牌别名测试类 - Brand Alias Test Class"""

    @pytest.fixture
    def toolset(self):
        """工具集"""
        repo = SQLitePartsRepository(DB_PATH)
        return Toolset(repo)

    def test_intel_chinese_alias(self, toolset):
        """测试 Intel 中文名匹配"""
        tools = toolset.register()
        search_parts = tools["search_parts"]
        
        # 使用英文 "Intel" 搜索，应该能匹配到中文 "英特尔" 品牌
        result = search_parts.invoke({
            "category": "cpu",
            "budget_max": 2000,
            "prefer_brands": ["Intel"],
        })
        
        print(f"\n[TEST] Search 'Intel' result count: {len(result)}")
        
        brands_in_result = set(p['brand'] for p in result)
        print(f"[TEST] Brands found: {brands_in_result}")
        
        # 应该包含 Intel 或 英特尔 品牌
        assert len(result) > 0, "应该能找到 Intel CPU"
        
        # 验证品牌匹配
        for p in result:
            assert p['brand'].lower() in ("intel", "英特尔"), \
                f"品牌应该是 Intel 或 英特尔，但找到: {p['brand']}"

    def test_amd_alias(self, toolset):
        """测试 AMD 别名匹配"""
        tools = toolset.register()
        search_parts = tools["search_parts"]
        
        result = search_parts.invoke({
            "category": "cpu",
            "budget_max": 2000,
            "prefer_brands": ["AMD"],
        })
        
        print(f"\n[TEST] Search 'AMD' result count: {len(result)}")
        
        assert len(result) > 0, "应该能找到 AMD CPU"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
