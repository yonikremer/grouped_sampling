from flask.testing import FlaskClient


def test_base_html(client: FlaskClient, auth) -> None:
    """Tests base.html that html files that extend base.html behave as expected."""
    first_response = client.get('/')
    assert first_response.status_code == 200
    lower_html_page = first_response.get_data(as_text=True).lower()
    lower_html_nav = lower_html_page.split('<nav>')[1].split('</nav>')[0]
    assert ("log in" in lower_html_nav or "login" in lower_html_nav)
    assert ("register" in lower_html_nav)
    assert not ("log out" in lower_html_nav or "logout" in lower_html_nav)

    auth.login()
    second_response = client.get('/')
    assert second_response.get_data(as_text=True) != first_response.get_data(as_text=True)
    assert second_response.status_code == 200
    lower_html_page = second_response.get_data(as_text=True).lower()
    lower_html_nav = lower_html_page.split('<nav>')[1].split('</nav>')[0]
    assert ("log out" in lower_html_nav or "logout" in lower_html_nav)
    assert not ("log in" in lower_html_nav or "login" in lower_html_nav)
    assert not ("register" in lower_html_nav)
