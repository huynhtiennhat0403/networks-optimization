"""
Scenarios Router
Handles scenario management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
import logging

from models.response_models import ScenarioListResponse, ScenarioInfo
from services.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(
    prefix="/scenarios",
    tags=["Scenarios"],
    responses={404: {"description": "Not found"}}
)

# Dependencies
_scenario_manager = None


def set_dependencies(scenario_manager):
    """Set global dependencies (called from main.py)"""
    global _scenario_manager
    _scenario_manager = scenario_manager


def get_scenario_manager():
    """Dependency to get scenario manager"""
    if _scenario_manager is None:
        raise HTTPException(status_code=503, detail="Scenario manager not initialized")
    return _scenario_manager


# ==================== ENDPOINTS ====================

@router.get("/", response_model=ScenarioListResponse)
@router.get("/list", response_model=ScenarioListResponse)
async def list_scenarios(
    category: Optional[str] = Query(None, description="Filter by category: home, office, public, mobile"),
    scenario_mgr: ScenarioManager = Depends(get_scenario_manager)
):
    """
    Get list of all available scenarios
    
    Optionally filter by category to narrow down results.
    """
    try:
        # Get all scenarios
        scenarios = scenario_mgr.get_all_scenarios()
        
        # Filter by category if specified
        if category:
            scenarios = [s for s in scenarios if s.get('category') == category.lower()]
            if not scenarios:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No scenarios found in category '{category}'"
                )
        
        # Get unique categories
        categories = list(set(s.get('category', 'other') for s in scenarios))
        
        # Convert to response format
        scenario_infos = [
            ScenarioInfo(
                id=s['id'],
                name=s['name'],
                description=s['description'],
                category=s.get('category', 'other'),
                expected_quality=s.get('expected_quality', 'Unknown'),
                parameters=None  # Don't include full params in list view
            )
            for s in scenarios
        ]
        
        logger.info(f"[SCENARIOS] Listed {len(scenario_infos)} scenarios (category: {category or 'all'})")
        
        return ScenarioListResponse(
            count=len(scenario_infos),
            scenarios=scenario_infos,
            categories=sorted(categories)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SCENARIOS] Error listing scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{scenario_id}", response_model=ScenarioInfo)
async def get_scenario(
    scenario_id: int,
    include_parameters: bool = Query(True, description="Include full network parameters"),
    scenario_mgr: ScenarioManager = Depends(get_scenario_manager)
):
    """
    Get details of a specific scenario by ID
    
    Returns complete scenario information including network parameters.
    """
    try:
        logger.info(f"[SCENARIOS] Fetching scenario ID: {scenario_id}")
        
        scenario = scenario_mgr.get_scenario(scenario_id)
        if not scenario:
            raise HTTPException(
                status_code=404, 
                detail=f"Scenario {scenario_id} not found"
            )
        
        # Prepare response
        response = ScenarioInfo(
            id=scenario['id'],
            name=scenario['name'],
            description=scenario['description'],
            category=scenario.get('category', 'other'),
            expected_quality=scenario.get('expected_quality', 'Unknown'),
            parameters=scenario['parameters'] if include_parameters else None
        )
        
        logger.info(f"[SCENARIOS] Returned scenario: {scenario['name']}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SCENARIOS] Error getting scenario: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category/{category}")
async def get_scenarios_by_category(
    category: str,
    scenario_mgr: ScenarioManager = Depends(get_scenario_manager)
):
    """
    Get all scenarios in a specific category
    
    Categories: home, office, public, mobile
    """
    try:
        scenarios = scenario_mgr.get_scenarios_by_category(category.lower())
        
        if not scenarios:
            raise HTTPException(
                status_code=404, 
                detail=f"No scenarios found in category '{category}'"
            )
        
        # Convert to response format (without full parameters)
        scenario_infos = [
            {
                "id": s['id'],
                "name": s['name'],
                "description": s['description'],
                "expected_quality": s.get('expected_quality', 'Unknown')
            }
            for s in scenarios
        ]
        
        logger.info(f"[SCENARIOS] Returned {len(scenario_infos)} scenarios for category '{category}'")
        
        return {
            "category": category,
            "count": len(scenario_infos),
            "scenarios": scenario_infos
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SCENARIOS] Error getting scenarios by category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/")
async def search_scenarios(
    q: str = Query(..., min_length=2, description="Search query (min 2 characters)"),
    scenario_mgr: ScenarioManager = Depends(get_scenario_manager)
):
    """
    Search scenarios by name or description
    
    Returns scenarios matching the search query.
    """
    try:
        scenarios = scenario_mgr.search_scenarios(q)
        
        if not scenarios:
            return {
                "query": q,
                "count": 0,
                "scenarios": [],
                "message": f"No scenarios found matching '{q}'"
            }
        
        # Convert to response format
        scenario_infos = [
            {
                "id": s['id'],
                "name": s['name'],
                "description": s['description'],
                "category": s.get('category', 'other'),
                "expected_quality": s.get('expected_quality', 'Unknown')
            }
            for s in scenarios
        ]
        
        logger.info(f"[SCENARIOS] Search '{q}' returned {len(scenario_infos)} results")
        
        return {
            "query": q,
            "count": len(scenario_infos),
            "scenarios": scenario_infos
        }
        
    except Exception as e:
        logger.error(f"[SCENARIOS] Error searching scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))