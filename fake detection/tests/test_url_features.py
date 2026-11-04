import pytest

from src.url_model import (URL_FEATURES, lexical_features, normalize_url, page_features,
                           registered_domain)


def test_normalize_and_errors():
    assert normalize_url("example.com") == "https://example.com"
    with pytest.raises(ValueError):
        normalize_url("")
    with pytest.raises(ValueError):
        normalize_url("ftp://example.com")


def test_registered_domain_handles_second_level_tlds():
    assert registered_domain("www.shop.example.co.uk") == "example.co.uk"
    assert registered_domain("a.b.example.com") == "example.com"


def test_lexical_legit_vs_suspicious():
    ok = lexical_features("https://www.example.com/")
    assert ok["having_IP_Address"] == 1 and ok["Prefix_Suffix"] == 1 and ok["having_Sub_Domain"] == 1
    bad = lexical_features("http://192.168.1.5:8080/secure-login@bank.verify.account.example.com//x/aaaaaaaaaaaaaaaaaaaa")
    assert bad["having_IP_Address"] == -1 and bad["port"] == -1 and bad["having_At_Symbol"] == -1
    assert bad["double_slash_redirecting"] == -1 and bad["URL_Length"] == -1
    assert lexical_features("https://bit.ly/abc")["Shortining_Service"] == -1


def test_page_features_detect_phishy_html():
    html = ('<form action="about:blank"><input></form><iframe src="x"></iframe>'
            '<a href="#">a</a><a href="javascript:void(0)">b</a><a href="https://evil.test/x">c</a>'
            '<img src="https://evil.test/a.png"><img src="https://evil.test/b.png">'
            '<link rel="icon" href="https://evil.test/f.ico">')
    f = page_features(html, "https://mybank.com/", redirects=5)
    assert f["SFH"] == -1 and f["Iframe"] == -1 and f["URL_of_Anchor"] == -1
    assert f["Request_URL"] == -1 and f["Favicon"] == -1 and f["Redirect"] == -1


def test_page_features_clean_html():
    html = '<a href="/about">a</a><img src="/logo.png"><form action="/login"></form>'
    f = page_features(html, "https://mybank.com/", redirects=0)
    assert all(f[k] == 1 for k in ("SFH", "Iframe", "URL_of_Anchor", "Request_URL", "Redirect"))


def test_all_features_covered():
    assert set(URL_FEATURES) >= {"SSLfinal_State", "DNSRecord", "Favicon", "Iframe"}