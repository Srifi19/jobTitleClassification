from dataclasses import dataclass
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from random import choice
from bs4 import BeautifulSoup

from helpers.logger import LOGGER, LoggingLevels


@dataclass
class ScrapedCourse:
    title: str
    description: str
    url: str


class CoursesScraper:
    def __init__(self, headless: bool = False) -> None:
        LOGGER.log("Initializing Courses Scraper...", level=LoggingLevels.INFO)
        self._HEADLESS = headless
        self._driver = None

    def _init_driver(self):
        self._driver = Chrome(
            service=Service(ChromeDriverManager().install()),
            options=_custom_secure_options(self._HEADLESS),
        )

    def get_top_courses(self, topic: str) -> list[ScrapedCourse]:
        # open the webpage after generating a corresponding url
        try:
            self._driver.get(_format_class_central_url(topic))
        except:
            self._init_driver()
            return self.get_top_courses(topic=topic)
        # get the page source
        page_source = self._driver.page_source
        # parse page source to extract top courses
        self._driver.close()
        return _parse_class_central_page(page_source)

    def reset_driver(self):
        try:
            # close current driver
            self._driver.close()
        except:
            pass
        finally:
            # re-init driver
            self._init_driver()


# ======================================= #
# --------------- Helpers --------------- #
# ======================================= #
# generate custom driver option
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/60.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/16.16299",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:70.0) Gecko/20100101 Firefox/70.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.4 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36 Edg/96.0.1054.34",
    "Mozilla/5.0 (Linux; Android 10; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.127 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36 Edge/18.19582",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36",
]


def _custom_secure_options(headless: bool = False) -> Options:
    options = Options()

    options.ignore_zoom_level = True
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Adding argument to disable the AutomationControlled flag
    options.add_argument("--disable-blink-features=AutomationControlled")

    # Exclude the collection of enable-automation switches
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    # Turn-off userAutomationExtension
    options.add_experimental_option("useAutomationExtension", False)

    preferences = {
        "profile.default_content_settings.geolocation": 2,
        "profile.managed_default_content_settings.images": 2,
    }

    options.add_experimental_option("prefs", preferences)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-blink-features")
    options.add_argument("--no-sandbox")
    # randomize user agent
    options.add_argument(f"user-agent={choice(USER_AGENTS)}")

    # enable headless mode
    if headless:
        options.add_argument("--headless")

    return options


# generate corresponding url based on a given topic
def _format_class_central_url(topic: str):
    topic = topic.strip().replace(" ", "+")
    return f"https://www.classcentral.com/search?q={topic}&free=true"


# parse class central web page and return all courses in the page
def _parse_class_central_page(web_page: str) -> list[ScrapedCourse]:
    # courseCentral [course] widget class
    COURSE_CONTAINER = "bg-white border-all border-gray-light padding-xsmall radius-small margin-bottom-small medium-up-padding-horz-large medium-up-padding-vert-medium course-list-course"
    soup = BeautifulSoup(web_page, "html.parser")

    # find all course containers
    course_containers = soup.find_all("li", class_=COURSE_CONTAINER)

    # results
    results = []
    for item in course_containers:
        try:
            title = item.find("h2").text.strip()
            description = item.find("p").text.strip()
            href = item.find("p").find("a")["href"]

            results.append(
                ScrapedCourse(
                    title=title,
                    description=description,
                    url=f"https://www.classcentral.com{href}",
                )
            )
        except Exception as e:
            continue

    return results
