# app/services/dashboard_service.py
from typing import Dict
import re

from fastapi import HTTPException, status
from neo4j.exceptions import ServiceUnavailable, Neo4jError

from app.core.neo4j_conn import get_driver
from app.helper import _dbname


async def get_basic_counts(organization: str, project: str) -> Dict[str, int]:
    dbname = _dbname(organization, project)

    try:
        driver = await get_driver()
    except Exception as e:
        # Neo4j driver cannot be created (connection refused / bad credentials)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j connection failed: {str(e)}"
        )

    cypher = """
    MATCH (b:Bug)
    WITH count(b) AS bug_count
    MATCH (d:Developer)
    WITH bug_count, count(d) AS developer_count
    MATCH (c:Commit)
    RETURN bug_count, developer_count, count(c) AS commit_count
    """

    try:
        async with driver.session(database=dbname) as session:
            result = await session.run(cypher)
            record = await result.single()
    except ServiceUnavailable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Neo4j unavailable: {str(e)}"
        )
    except Neo4jError as e:
        # Database does not exist or other cypher error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Neo4j query error: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error querying Neo4j: {str(e)}"
        )

    if record is None:
        # Database exists but empty
        return {
            "bug_count": 0,
            "developer_count": 0,
            "commit_count": 0,
        }

    return {
        "bug_count": record["bug_count"],
        "developer_count": record["developer_count"],
        "commit_count": record["commit_count"],
    }
