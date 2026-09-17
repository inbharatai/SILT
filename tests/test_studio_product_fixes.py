"""Product regressions: actual JS/Chromium and safe GETs; no model jobs.

Uses the optional developer Playwright installation (no frontend dependency).
The reader is a bounded Markdown subset, not a CommonMark implementation.
All server roots are temporary; readiness tests forbid manager/probe access.
"""
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'src/asea/studio/static/index.html'


def isolated_env(tmp_path, flag=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
               PRODUCT_RUNTIME=str(tmp_path), SILT_EXPERIMENTAL_ROOT=str(tmp_path / 'experimental'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
    env.pop('ASEA_RUN_REAL', None)
    env.pop('SILT_ENABLE_EXPERIMENTAL', None)
    if flag is not None:
        env['SILT_ENABLE_EXPERIMENTAL'] = flag
    return env


@pytest.mark.parametrize('flag', [None, '0', 'true', '1'])
def test_health_code_availability_exact_startup_and_current_flag(tmp_path, flag):
    # Fresh interpreter is essential: routes mount at server import time. No
    # reload leaks or reuse of a previous parameter's mounted app.
    script = r'''
import json, os, sys
from pathlib import Path
from asea.studio import jobs
jobs.ROOT=Path(os.environ['PRODUCT_RUNTIME'])
from asea.studio import server
from fastapi.testclient import TestClient
mounted=os.environ.get('SILT_ENABLE_EXPERIMENTAL')=='1'
client=TestClient(server.app,base_url='http://127.0.0.1')
if mounted:
    from asea.studio import experimental as ex
    def forbidden(): raise AssertionError('health instantiated experimental manager')
    ex.manager=forbidden
before_files=sorted(str(p) for p in jobs.ROOT.rglob('*'))
before_modules=set(sys.modules)
reports=[]
for value in [os.environ.get('SILT_ENABLE_EXPERIMENTAL'), '0', 'true', None, '1']:
    if value is None: os.environ.pop('SILT_ENABLE_EXPERIMENTAL',None)
    else: os.environ['SILT_ENABLE_EXPERIMENTAL']=value
    response=client.get('/api/health');assert response.status_code==200
    h=response.json()
    assert h['ok'] is True and h['service']=='silt-studio' and h['mock_free'] is True
    assert h['experimental']==dict(code_available=True,routes_mounted=mounted,enabled=mounted and value=='1')
    # Off-after-on is enforced by the actual experimental request guard. An
    # off-start changed to 1 still has no routes until a fresh server starts.
    if not (mounted and value=='1'):
        assert client.get('/experimental').status_code==404
        assert client.get('/api/experimental/schema').status_code==404
    reports.append(h['experimental'])
assert before_files==sorted(str(p) for p in jobs.ROOT.rglob('*'))
assert not any(n.split('.')[0] in {'torch','transformers','peft','sentence_transformers'} for n in set(sys.modules)-before_modules)
assert not (jobs.ROOT/'experimental').exists()
assert not server.manager.jobs
print(json.dumps(reports))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env=isolated_env(tmp_path, flag), text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 5


@pytest.fixture(scope='module')
def browser():
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--no-sandbox'])
        yield browser
        browser.close()


@pytest.fixture(scope='module', params=[None, '1'], ids=['off', 'on'])
def live_studio(request, tmp_path_factory):
    runtime = tmp_path_factory.mktemp('product-studio')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = isolated_env(runtime, request.param)
    env['PRODUCT_PORT'] = str(port)
    script = r'''
import os
from pathlib import Path
from asea.studio import jobs
jobs.ROOT=Path(os.environ['PRODUCT_RUNTIME'])
from asea.studio import server
from fastapi.responses import FileResponse
@server.app.get('/__product_public_setup')
def public_setup():
    return FileResponse(Path(server.__file__).resolve().parents[3]/'docs/studio/index.html')
import uvicorn
uvicorn.run(server.app,host='127.0.0.1',port=int(os.environ['PRODUCT_PORT']),log_level='error',access_log=False)
'''
    proc = subprocess.Popen([sys.executable, '-c', script], cwd=runtime, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f'http://127.0.0.1:{port}'
    try:
        for _ in range(150):
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            try:
                with urllib.request.urlopen(url + '/api/health', timeout=.3):
                    break
            except OSError:
                time.sleep(.1)
        else:
            pytest.fail('Studio startup timeout')
        yield url, request.param
        for path, key in [('/api/transfers', 'jobs'), ('/api/deepapply', 'jobs'), ('/api/spring', 'jobs')]:
            with urllib.request.urlopen(url + path) as response:
                assert not json.load(response)[key]
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        proc.stderr.close()


def safe_page(browser, live_studio, width):
    url, _ = live_studio
    page = browser.new_page(viewport={'width': width, 'height': 1000 if width == 1440 else 844})
    bad = []
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    def route(r):
        if r.request.method not in ('GET', 'HEAD') or not r.request.url.startswith(url + '/'):
            bad.append((r.request.method, r.request.url))
            r.abort()
        else:
            r.continue_()
    page.route('**/*', route)
    page.goto(url, wait_until='networkidle')
    page.wait_for_function("document.querySelector('#sender').options.length > 0")
    return page, bad, errors


@pytest.mark.parametrize('width', [1440, 390])
def test_all_tabs_scoped_mobile_forms_public_setup_and_discovery(browser, live_studio, width):
    page, bad, errors = safe_page(browser, live_studio, width)
    try:
        assert page.locator('.tab[data-tab]').count() == 8
        assert page.locator('#connStatus').inner_text() == 'App available'
        assert 'quality are unchecked' in page.locator('#connMsg').inner_text()
        link = page.locator('#experimental-link')
        if live_studio[1] == '1':
            assert link.is_visible() and link.get_attribute('href') == '/experimental'
        else:
            assert not link.is_visible() and link.get_attribute('href') is None
            assert 'disabled' in page.locator('#experimental-status').inner_text()
        page.locator('#experimental-instructions > summary').click()
        assert 'SILT_ENABLE_EXPERIMENTAL=1' in page.locator('#experimental-help').inner_text()
        page.locator('#experimental-instructions > summary').click()
        if live_studio[1] == '1':
            link.click()
            page.wait_for_url(live_studio[0] + '/experimental')
            page.wait_for_load_state('networkidle')
            assert page.locator('#jobs').is_visible()
            page.go_back(wait_until='networkidle')
            page.wait_for_function("document.querySelector('#sender').options.length > 0")
        for tab in page.locator('.tab[data-tab]').all():
            name = tab.get_attribute('data-tab')
            tab.click()
            assert page.locator('#tab-' + name).is_visible()
            if name in ('train', 'compress'):
                for detail in page.locator('#tab-' + name + ' details').all():
                    detail.evaluate('(e)=>e.open=true')
                controls = page.locator('#tab-' + name + ' input, #tab-' + name + ' select, #tab-' + name + ' button')
                for control in controls.all():
                    if control.is_visible():
                        rect = control.bounding_box()
                        assert rect['x'] >= 0 and rect['x'] + rect['width'] <= width + 1, (name, control.get_attribute('id'), rect)
            assert page.evaluate('document.documentElement.scrollWidth') <= width + 1, name
        for purpose, target in [('transfer', 'transfer'), ('train', 'train'), ('compress', 'compress')]:
            page.locator('[data-workflow="' + purpose + '"]').click()
            assert page.locator('#tab-' + target).is_visible()
        page.locator('#model-readiness > summary').click()
        assert page.locator('#selected-model-readiness li').count() == 4
        assert 'not verified' in page.locator('#selected-model-readiness').inner_text()
        # Switch catalog selection only; no model probe/job/download.
        cloud_options = page.locator('#sender option').evaluate_all("es=>es.filter(e=>/tts/i.test(e.textContent)).map(e=>e.value)")
        if cloud_options:
            page.locator('[data-workflow="transfer"]').click()
            page.locator('#sender').select_option(cloud_options[0])
            assert 'CLOUD-BACKED' in page.locator('#selected-model-readiness').inner_text()
        page.goto(live_studio[0] + '/__product_public_setup', wait_until='networkidle')
        assert page.evaluate('document.documentElement.scrollWidth') <= width + 1
        assert 'disabled by default' in page.locator('#experimental-setup').inner_text()
        assert page.locator('script').count() == 0  # no public-to-local token/probe bridge
        assert not bad and not errors, (bad, errors)
    finally:
        page.close()


def source_tables():
    lines = (ROOT / 'README.md').read_text().splitlines()
    tables = []
    for i, line in enumerate(lines[:-1]):
        if '|' in line and re.fullmatch(r'[\s|:\-]+', lines[i + 1]) and '-' in lines[i + 1]:
            rows = [line]
            j = i + 2
            while j < len(lines) and '|' in lines[j] and lines[j].strip():
                rows.append(lines[j])
                j += 1
            tables.append(rows)
    return tables


@pytest.mark.parametrize('width', [1440, 390])
def test_live_readme_all_18_first_rows_16_portfolio_6_final_arms(browser, live_studio, width):
    page, bad, errors = safe_page(browser, live_studio, width)
    try:
        page.locator('#readme > summary').click()
        page.wait_for_selector('#readme-body table')
        tables = source_tables()
        assert len(tables) == 18
        rendered = page.locator('#readme-body table')
        assert rendered.count() == len(tables)
        # Independent source normalization for this README's simple table
        # syntax: never use the JS renderer as its own expected-value oracle.
        def plain(cell):
            cell = re.sub(r'`([^`]+)`', r'\1', cell)
            cell = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cell)
            cell = cell.replace('**', '')
            return re.sub(r'\*([^*]+)\*', r'\1', cell).strip()
        expected = [[[plain(c) for c in row.strip().strip('|').split('|')]
                     for row in rows] for rows in tables]
        actual = rendered.evaluate_all(
            'ts=>ts.map(t=>[...t.rows].map(r=>[...r.cells].map(c=>c.textContent.trim())))')
        assert sum(len(row) for table in expected for row in table) == 501
        assert [len(t) - 1 for t in expected] == [16, 6, 6, 3, 14, 16, 11, 3, 3, 6, 4, 5, 3, 3, 6, 22, 12, 6]
        assert actual == expected
        with urllib.request.urlopen(live_studio[0] + '/api/readme') as response:
            assert response.read() == (ROOT / 'README.md').read_bytes()
        assert rendered.nth(0).locator('tbody tr').count() == 16
        assert 'Core L3' in rendered.nth(0).locator('tbody tr').first.inner_text()
        final = page.locator('#readme-body table').filter(has_text='Frozen final arm')
        assert final.locator('tbody tr').count() == 6
        assert final.locator('tbody tr').first.locator('td').nth(1).inner_text() == '12'
        assert page.locator('#readme .md-link-off').count() == 0
        links = page.locator('#readme-body a').evaluate_all('es=>es.map(e=>({href:e.getAttribute("href"),rel:e.rel,target:e.target}))')
        assert len(links) >= 88
        # The original 86 inert references include internal anchors, not just
        # repository paths. Restored first rows add additional working links.
        assert sum(l['href'].startswith(('https://github.com/inbharatai/SILT/blob/main/', '#readme-')) for l in links) >= 86
        for link in links:
            if link['href'].startswith('#readme-'):
                assert page.locator(link['href']).count() == 1, link
        for link in links:
            assert link['href'].startswith(('https://', '#readme-'))
            if link['href'].startswith('https://'):
                assert link['target'] == '_blank' and 'noopener' in link['rel'] and 'noreferrer' in link['rel']
        assert page.locator('#readme-body h1').first.inner_text() == 'SILT — Skill Interchange Layer with Trust-gating'
        assert '<p align=' not in page.locator('#readme-body').inner_text()
        assert page.locator('#readme-body img, #readme-body script').count() == 0
        assert page.locator('#readme a[href="https://github.com/inbharatai/SILT/blob/main/README.md"]').count() >= 1
        assert not bad and not errors, (bad, errors)
    finally:
        page.close()


def test_real_js_renderer_security_and_general_table_cases(browser, live_studio):
    page, bad, errors = safe_page(browser, live_studio, 1440)
    try:
        hostile = ['javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'data:text/html,test', 'vbscript:foo',
                   '//evil.test/x', '/api/readme', '\\evil.test', 'https:\\evil.test',
                   'java\tscript:alert(1)', 'https://safe.test/\nattack', '%6aavascript:foo',
                   'java%0ascript:foo', '%252f%252fevil.test', '&#106;avascript:foo',
                   'javascript&colon;foo', 'https://user:pass@evil.test', '../outside.md',
                   '%2e%2e/outside.md', 'file:///tmp/private', 'http://unsafe.test',
                   'https://safe.test/%00bad', '\u0085javascript:foo',
                   'java\u200bscript:foo', 'https://safe.test/%E2%80%AEbad']
        assert page.evaluate('urls=>urls.map(readmeHref)', hostile) == [None] * len(hostile)
        assert page.evaluate("readmeHref('docs/CAPABILITIES.md')") == 'https://github.com/inbharatai/SILT/blob/main/docs/CAPABILITIES.md'
        assert page.evaluate("readmeHref('./LOCAL_SETUP.md')") == 'https://github.com/inbharatai/SILT/blob/main/LOCAL_SETUP.md'
        markdown = '''# Hello World
[jump](#hello-world)
# Hello World
| Header | Value |
| --- | --- |
| first | `a|b` |
| second | a\\|b |

A | B
--- | ---
only | body

| Empty |
| --- |

<script>window.__readerXSS=1</script>
<img src=x onerror="window.__readerXSS=1">
[x](javascript:alert(1)) [x](data:text/html,evil) [x](//evil.test)
[x](https://safe.test/?a=1&b=2)
`[not a link](https://evil.test)`
`<script>literal code</script>`
```html
<p align="center">literal fenced markup</p>
```
[<svg/onload=window.__readerXSS=1>](https://safe.test/)
'''
        result = page.evaluate('''md=>{
            const d=document.createElement('section');d.id='reader-fixture';document.body.append(d);
            d.innerHTML=renderMarkdown(md);
            return {rows:[...d.querySelectorAll('table')].map(t=>[...t.querySelectorAll('tbody tr')].map(r=>[...r.cells].map(c=>c.textContent))),
              headings:[...d.querySelectorAll('h1')].map(e=>e.id), links:[...d.querySelectorAll('a')].map(e=>e.getAttribute('href')),
              active:d.querySelectorAll('script,img,svg,iframe,object').length,
              fenced:d.querySelector('pre code').textContent, code:[...d.querySelectorAll('code')].map(e=>e.textContent)};
        }''', markdown)
        assert result['rows'] == [[['first', 'a|b'], ['second', 'a|b']], [['only', 'body']], []]
        assert result['headings'] == ['readme-hello-world', 'readme-hello-world-1']
        assert result['active'] == 0
        assert '#readme-hello-world' in result['links']
        assert 'https://safe.test/?a=1&b=2' in result['links']
        assert not any('evil.test' in l or l.startswith(('data:', 'javascript:', '//')) for l in result['links'])
        assert '[not a link](https://evil.test)' in result['code']
        assert '<script>literal code</script>' in result['code']
        assert '<p align="center">literal fenced markup</p>' in result['fenced']
        page.locator('#reader-fixture a[href="#readme-hello-world"]').click()
        assert page.evaluate('location.hash') == '#readme-hello-world'
        assert page.evaluate('window.__readerXSS || 0') == 0
        assert not bad and not errors, (bad, errors)
    finally:
        page.close()


@pytest.fixture
def renderer_page(browser):
    # Execute the shipped JavaScript in Chromium without bootstrapping the app.
    # No regex rewrite/reimplementation of the renderer, no network or server.
    page = browser.new_page()
    page.route('**/*', lambda route: route.abort())
    source = UI.read_text()
    page.add_script_tag(content=source[source.index('const README_REPO_BASE='):
                                       source.index('let _readmeLoaded=false;')])
    page.set_content('<section id="fixture"></section>')
    yield page
    page.close()


def rendered_fixture(page, markdown):
    return page.evaluate(r'''md=>{
        window.__edgeXSS=0;
        const d=document.querySelector('#fixture'); d.innerHTML=renderMarkdown(md);
        return {
          ids:[...d.querySelectorAll('[id]')].map(e=>e.id),
          headings:[...d.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(e=>[e.textContent,e.id]),
          links:[...d.querySelectorAll('a')].map(e=>[e.textContent,e.getAttribute('href')]),
          rows:[...d.querySelectorAll('table')].map(t=>[...t.rows].map(r=>[...r.cells].map(c=>c.textContent))),
          code:[...d.querySelectorAll('code')].map(e=>e.textContent), text:d.textContent,
          active:d.querySelectorAll('script,img,svg,iframe,object,embed').length,
          handlers:[...d.querySelectorAll('*')].flatMap(e=>[...e.attributes].filter(a=>/^on/i.test(a.name))).length,
          xss:window.__edgeXSS
        };
    }''', markdown)


@pytest.mark.parametrize('markdown,ids,links', [
    ('[forward](#a-1)\n# A\n# A-1\n# A\n[jump](#a-1)\n[repeat](#a-2)',
     ['readme-a', 'readme-a-1', 'readme-a-2'],
     [['forward', '#readme-a-1'], ['jump', '#readme-a-1'], ['repeat', '#readme-a-2']]),
    ('[forward](#a-1)\n# A\n# A\n> # A-1\n> # A\n\n# A-1\n[jump](#a-1)',
     ['readme-a', 'readme-a-1', 'readme-a-1-1', 'readme-a-2', 'readme-a-1-2'],
     [['forward', '#readme-a-1-1'], ['jump', '#readme-a-1-1']]),
    ('# A\n> # A-1\n> > # A\n\n- item\n  # A-2\n  # A\n\n# A\n[jump](#a)',
     ['readme-a', 'readme-a-1', 'readme-a-2', 'readme-a-2-1', 'readme-a-3', 'readme-a-4'],
     [['jump', '#readme-a']]),
    ('# !!!\n# -1\n# ???\n# __proto__\n# __proto__\n[x](#__proto__)',
     ['readme-', 'readme--1', 'readme--2', 'readme-__proto__', 'readme-__proto__-1'],
     [['x', '#readme-__proto__']]),
    ('[forward](#caf%C3%A9)\n# Café\n# Café\n[repeat](#caf%C3%A9-1)',
     ['readme-café', 'readme-café-1'],
     [['forward', '#readme-caf%C3%A9'], ['repeat', '#readme-caf%C3%A9-1']]),
])
def test_heading_collision_set_and_first_heading_anchor_map(renderer_page, markdown, ids, links):
    result = rendered_fixture(renderer_page, markdown)
    assert result['ids'] == ids
    assert len(set(ids)) == len(ids)
    assert result['links'] == links
    assert all(i.startswith('readme-') for i in ids)
    assert rendered_fixture(renderer_page, markdown) == result  # no leaked state
    assert renderer_page.evaluate('''()=>[...document.querySelectorAll('#fixture a')].every(a=>
        document.getElementById(decodeURIComponent(a.hash.slice(1))))''')
    for label, href in links:
        renderer_page.get_by_role('link', name=label, exact=True).click()
        assert renderer_page.evaluate('location.hash') == href


@pytest.mark.parametrize('slashes', range(1, 7))
def test_pipe_backslash_parity_including_optional_trailing_border(renderer_page, slashes):
    source = 'a' + '\\' * slashes + '|b'
    expected = ['a' + '\\' * (slashes // 2) + '|b'] if slashes % 2 else ['a' + '\\' * (slashes // 2), 'b']
    for border in ['', '|']:
        result = rendered_fixture(renderer_page, '| H | V |\n|---|---|\n|' + source + border)
        assert result['rows'] == [[['H', 'V'], expected]]
    # A final escaped pipe is cell text, not a syntactic border; an even
    # backslash run leaves a real border. Empty interior cells stay present.
    result = rendered_fixture(renderer_page, '| H |\n|---|\n|a' + '\\' * slashes + '|')
    expected_last = 'a' + '\\' * (slashes // 2) + ('|' if slashes % 2 else '')
    assert result['rows'] == [[['H'], [expected_last]]]


@pytest.mark.parametrize('body,expected,code', [
    ('|a|`x``|y|', ['a', '`x``', 'y'], []),
    ('|a|``x`|y|', ['a', '``x`', 'y'], []),
    ('|a|`x``|y`|z|', ['a', 'x``|y', 'z'], ['x``|y']),
    ('|a|``x`|y``|z|', ['a', 'x`|y', 'z'], ['x`|y']),
    ('|a|```x``|y```|z|', ['a', 'x``|y', 'z'], ['x``|y']),
    ('|a|````x```|y````|z|', ['a', 'x```|y', 'z'], ['x```|y']),
    (r'|a|\`x|y`|z|', ['a', '`x', 'y`', 'z'], []),
    ('|a|`x|y|z|', ['a', '`x', 'y', 'z'], []),
    ('||a||b||', ['', 'a', '', 'b', ''], []),
])
def test_table_exact_backtick_runs_and_unmatched_literal_delimiters(renderer_page, body, expected, code):
    result = rendered_fixture(renderer_page, '| H | V |\n|---|---|\n' + body)
    assert result['rows'] == [[['H', 'V'], expected]]
    assert result['code'] == code


@pytest.mark.parametrize('rows', [[], ['|first|1|'], ['|first|1|', '|second|2|', '|third|3|']])
def test_table_zero_one_multiple_rows_do_not_drop_first(renderer_page, rows):
    result = rendered_fixture(renderer_page, '\n'.join(['| H | V |', '|---|---|'] + rows))
    assert result['rows'] == [[['H', 'V']] + [r.strip('|').split('|') for r in rows]]


def test_mismatched_inline_delimiters_escape_source_and_keep_urls_inert(renderer_page):
    markdown = r'''| H | V |
|---|---|
|a|`<svg/onload=window.__edgeXSS=1>``|[bad](javascript:window.__edgeXSS=1)|
|a|``<iframe srcdoc="<script>window.__edgeXSS=1</script>">`|[bad](data:text/html,evil)|

`x`` unmatched

``x` unmatched

`<script>window.__edgeXSS=1</script>`
``[not-link](https://evil.test)``
[x](%256aavascript:foo) [x](//evil.test) [x](../outside.md)
[<img src=x onerror=window.__edgeXSS=1>](https://safe.test/)
<svg onload="window.__edgeXSS=1"></svg>
&#60;img src=x onerror=window.__edgeXSS=1&#62;
```html
<script>window.__edgeXSS=1</script>
```
'''
    result = rendered_fixture(renderer_page, markdown)
    assert [len(row) for row in result['rows'][0]] == [2, 3, 3]
    assert result['rows'][0][1][1] == '`<svg/onload=window.__edgeXSS=1>``'
    assert result['active'] == result['handlers'] == result['xss'] == 0
    assert result['links'] == [['<img src=x onerror=window.__edgeXSS=1>', 'https://safe.test/']]
    assert result['code'] == ['<script>window.__edgeXSS=1</script>', '[not-link](https://evil.test)', '<script>window.__edgeXSS=1</script>']
    assert '`x`` unmatched' in result['text'] and '``x` unmatched' in result['text']
    assert renderer_page.evaluate('window.__edgeXSS') == 0
