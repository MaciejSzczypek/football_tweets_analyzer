from typing import Set, Optional, Tuple, Dict, FrozenSet
from urllib.parse import urljoin
import bs4
import requests
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Person:
    last_name: str
    team_name: str
    role: str
    first_name: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.first_name} {self.last_name} ({self.team_name}, {self.role})"

    def to_dict(self) -> Dict[str, str]:
        return vars(self)


@dataclass(frozen=True)
class TeamSquad:
    team_name: str
    persons: FrozenSet[Person]

    def find_person_in_squad(self, last_name: str) -> Optional[Person]:
        lowered_last_name = last_name.lower()
        for person in self.persons:
            if person.last_name.lower() == lowered_last_name:
                return person


@dataclass
class TeamsSquads:
    team_1: TeamSquad
    team_2: TeamSquad

    def find_person_in_squads(self, last_name: str) -> Optional[Person]:
        person_found_in_team_1 = self.team_1.find_person_in_squad(last_name=last_name)
        if person_found_in_team_1:
            return self.team_1.find_person_in_squad(last_name=last_name)
        return self.team_2.find_person_in_squad(last_name=last_name)


class TeamsSquadsScrapper:
    WORLD_FOOTBALL_WEBSITE_ROOT_URL = "https://www.worldfootball.net"
    WORLD_FOOTBALL_PREMIER_LEAGUE_19_20_TEAMS_URL = urljoin(
        WORLD_FOOTBALL_WEBSITE_ROOT_URL, "persons/eng-premier-league-2019-2020"
    )
    WORLD_FOOTBALL_PREMIER_LEAGUE_19_20_FOOTBALL_SQUAD_URL_SUFFIX = "2020/2/"

    def __init__(self, team_1_name: str, team_2_name: str) -> None:
        self._team_1_name = team_1_name
        self._team_2_name = team_2_name
        self._team_names = {team_1_name, team_2_name}

    def get_teams_squads(self) -> TeamsSquads:
        team_keys = self._get_team_names_to_keys_mapping(
            url=self.WORLD_FOOTBALL_PREMIER_LEAGUE_19_20_TEAMS_URL,
            team_names=self._team_names,
        )
        team_1_squad = self._get_team_squad(
            team_name=self._team_1_name, team_key=team_keys.get(self._team_1_name),
        )
        team_2_squad = self._get_team_squad(
            team_name=self._team_2_name, team_key=team_keys.get(self._team_2_name),
        )
        return TeamsSquads(team_1=team_1_squad, team_2=team_2_squad,)

    @classmethod
    def _get_team_squad(cls, team_name: str, team_key: str) -> TeamSquad:
        team_squad_url = urljoin(
            cls.WORLD_FOOTBALL_WEBSITE_ROOT_URL,
            f"{team_key}/{cls.WORLD_FOOTBALL_PREMIER_LEAGUE_19_20_FOOTBALL_SQUAD_URL_SUFFIX}",
        )
        page_content = cls._get_page_content(team_squad_url)
        parser = cls._get_parser(page_content)
        role = None
        persons = set()
        for row in parser.find_all("tr"):
            for row_child in row.children:
                if row_child.name == "th":
                    role = row_child.text
                elif row_child.name == "td" and role:
                    link_element = row_child.find("a")
                    if link_element:
                        (
                            person_first_name,
                            person_last_name,
                        ) = cls._get_person_name_from_link_element(link_element)
                        if person_last_name:
                            persons.add(
                                Person(
                                    last_name=person_last_name,
                                    first_name=person_first_name,
                                    team_name=team_name,
                                    role=role,
                                )
                            )
        return TeamSquad(team_name=team_name, persons=frozenset(persons),)

    @classmethod
    def _get_team_names_to_keys_mapping(
        cls, url: str, team_names: Set[str]
    ) -> Dict[str, str]:
        page_content = cls._get_page_content(url)
        parser = cls._get_parser(page_content)
        team_keys = {}
        for team_name in team_names:
            team_badge_element = parser.find("img", title=team_name)
            team_keys[team_name] = cls._get_team_key(
                team_badge_element=team_badge_element
            )
        return team_keys

    @classmethod
    def _get_team_key(cls, team_badge_element) -> str:
        return team_badge_element.parent.get("href").rsplit("/", 1)[0]

    @classmethod
    def _get_person_name_from_link_element(
        cls, link_element
    ) -> Optional[Tuple[str, str]]:
        first_name = None
        last_name = None
        href = link_element.get("href")
        if re.match(r"/player_summary/([a-zA-Z\-])+/", href):
            original_person_name = link_element.get("title")
            lowered_person_name_without_diacritics = cls._get_lowered_person_name_without_diacritics(
                href
            )
            person_name_without_diacritics = cls._transform_diacritics_to_ascii_characters(
                original_name=original_person_name,
                lowered_name_without_diacritics=lowered_person_name_without_diacritics,
            )
            name_tokens = person_name_without_diacritics.split()
            if len(name_tokens) > 1:
                first_name = name_tokens[0]
                last_name = " ".join(name_tokens[1:])
            else:
                last_name = name_tokens[0]

        return first_name, last_name

    @classmethod
    def _get_page_content(cls, url: str) -> str:
        page = requests.get(url)
        return page.content

    @classmethod
    def _get_parser(cls, page_content: str) -> bs4.BeautifulSoup:
        return bs4.BeautifulSoup(page_content, "html.parser")

    @classmethod
    def _get_lowered_person_name_without_diacritics(
        cls, person_summary_key: str,
    ):
        ascii_person_name_with_dashes = person_summary_key.split("/")[-2]
        return " ".join(ascii_person_name_with_dashes.split("-"))

    @classmethod
    def _transform_diacritics_to_ascii_characters(
        cls, original_name: str, lowered_name_without_diacritics: str
    ):
        new_name = ""
        for index, character in enumerate(original_name):
            if character.isascii():
                new_name += character
            else:
                character_in_ascii = lowered_name_without_diacritics[index]
                if character.isupper():
                    character_in_ascii.upper()
                new_name += character_in_ascii
        return new_name
