"""The upload UI is served and wired to the API it calls."""

from httpx import AsyncClient


async def test_index_is_served_at_root(client: AsyncClient) -> None:
    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Document QA</title>" in response.text


async def test_ui_posts_to_the_documented_endpoint_with_both_fields(
    client: AsyncClient,
) -> None:
    """Guards against the page and the API drifting apart."""
    page = (await client.get("/")).text

    assert '"/api/v1/qa"' in page
    assert 'body.append("questions"' in page
    assert 'body.append("document"' in page


async def test_ui_renders_all_three_result_states(client: AsyncClient) -> None:
    page = (await client.get("/")).text

    assert "Answered" in page
    assert "Not found" in page
    assert "Error: " in page, "a timeout must not be shown as a plain 'not found'"
