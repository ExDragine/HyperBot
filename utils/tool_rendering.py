"""Template helpers and request headers shared by tools that render images."""

from jinja2 import Environment, FileSystemLoader

from utils.markdown_render import TEMPLATES_DIR

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

API_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

IMG_HEADERS = {
    "User-Agent": USER_AGENT,
}


def load_css(filename: str) -> str:
    """读取 templates/ 下的 CSS 文件。"""
    return (TEMPLATES_DIR / filename).read_text(encoding="utf-8")


def render_template(name: str, **context) -> str:
    """用 templates/ 下的 Jinja2 模板渲染 HTML 片段。"""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True)
    return env.get_template(name).render(**context)
