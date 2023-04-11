""" Timex tokenizer for the added vocabulary.
Related to the multilingual temporal tagging (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH
@author: Lukas Lange

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from typing import List, Tuple


class HeideltimeValue(ABC):
    @abstractmethod
    def parseString(self, value: str):
        return

    @abstractmethod
    def reconstructValue(self, value_tokens: list[str], assertions: bool = False):
        return


class D1(HeideltimeValue):
    regex = (
        "(PRESENT|PAST|FUTURE)_REF" "(?:T(\d\d|X|MO|MD|EV|AF|NI|MI|DT)?(?::(\d\d))?)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f1 = m.group(1)
            f2 = "REF"
            f_time_1 = m.group(2)
            f_time_2 = m.group(3)
            return {
                "Basic": f1,
                "Date_1": f2,
                "Date_2": "[PAD]",
                "Date_3": "[PAD]",
                "Date_4": "[PAD]",
                "Time_1": f_time_1,
                "Time_2": f_time_2,
                "Time_3": "[PAD]",
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):
        if assertions:
            assert value_tokens[0] in ["PRESENT", "PAST", "FUTURE"]
            assert value_tokens[1] == "REF"
            assert value_tokens[2] == "[PAD]"
            assert value_tokens[3] == "[PAD]"
            assert value_tokens[4] == "[PAD]"
            assert value_tokens[7] == "[PAD]"
            assert value_tokens[8] == "[PAD]"
            assert value_tokens[9] == "[PAD]"
            assert value_tokens[10] == "[PAD]"

        basic = value_tokens[0]
        time_1 = value_tokens[5]
        time_2 = value_tokens[6]

        decoded_val = basic + "_REF"
        if time_1 != "[PAD]":
            decoded_val += "T" + time_1
            if time_2 != "[PAD]":
                decoded_val += ":" + time_2
        return decoded_val


class D2(HeideltimeValue):
    regex = (
        "(BC)?(\d\d?|XX)'?(\d\d|XX)?"
        "(?:-(W)?(\d\d?|XX|SP|SU|FA|AU|WI|H1|H2|Q1|Q2|Q3|Q4|H|Q|M1|M2))?"
        "(?:-(\d\d?|XX|WE))?\)?"
        "(?:T(\d\d|X|MO|MD|EV|AF|NI|MI|DT|XX)?(?::(\d\d))?(?:(?::|-)(\d\d))?)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_bc = m.group(1)
            f_year_1 = m.group(2)
            f_year_2 = m.group(3)
            f_week = m.group(4)
            f_month_season = m.group(5)
            f_day_weekend = m.group(6)
            f_time_1 = m.group(7)
            f_time_2 = m.group(8)
            f_time_3 = m.group(9)

            return {
                "Basic": f_bc,
                "Date_1": f_year_1,
                "Date_2": f_year_2,
                "Date_3": f_month_season,
                "Date_4": f_day_weekend,
                "Time_1": f_time_1,
                "Time_2": f_time_2,
                "Time_3": f_time_3 if f_time_3 else f_week,
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):

        if assertions:
            assert value_tokens[0] in ["[PAD]", "BC"]
            assert value_tokens[8] == "[PAD]"
            assert value_tokens[9] == "[PAD]"
            assert value_tokens[10] == "[PAD]"

        bc_ind = value_tokens[0]
        year_1 = value_tokens[1]
        year_2 = value_tokens[2]
        month_season = value_tokens[3]
        day_weekend = value_tokens[4]
        time_1 = value_tokens[5]
        time_2 = value_tokens[6]
        time_3 = value_tokens[7]

        decoded_val = "BC" if bc_ind == "BC" else ""
        decoded_val += year_1
        if year_2 != "[PAD]":
            decoded_val += year_2
        if month_season != "[PAD]":
            decoded_val += "-W" if time_3 == "W" else "-"
            decoded_val += month_season
            if day_weekend != "[PAD]":
                decoded_val += "-" + day_weekend
        if time_1 != "[PAD]":
            decoded_val += "T" + time_1
            if time_2 != "[PAD]":
                decoded_val += ":" + time_2
                if time_3 != "[PAD]" and time_3 != "W":
                    decoded_val += ":" + time_3
        return decoded_val


class P1(HeideltimeValue):
    regex = (
        "(P|PT)(?:(\d\d?|X|XX)(\d\d|\.)?(\d\d?)?)?"
        "(H|D|DE|DT|M|C|Y|C|CE|W|WE|Qu|Q|S|NI|AF|MO|EV|MD|MI)?"
        "(\d\d?)?(H|D|DE|DT|M|C|Y|C|CE|W|WE|Qu|Q|S|NI|AF|MO|EV|MD|MI)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_time = m.group(1)
            f_dur_1 = m.group(2)
            f_dur_2 = m.group(3)
            f_dur_3 = m.group(4)
            f_unit_1 = m.group(5)
            f_dur_4 = m.group(6)
            f_unit_2 = m.group(7)

            return {
                "Basic": f_time,
                "Date_1": f_dur_1,
                "Date_2": f_dur_2,
                "Date_3": f_dur_3,
                "Date_4": f_dur_4,
                "Time_1": f_unit_1,
                "Time_2": f_unit_2,
                "Time_3": "[PAD]",
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):
        if assertions:
            assert value_tokens[0] in ["P", "PT"]
            assert value_tokens[7] == "[PAD]"
            assert value_tokens[8] == "[PAD]"
            assert value_tokens[9] == "[PAD]"
            assert value_tokens[10] == "[PAD]"

        dur_ind = value_tokens[0]
        dur_1 = value_tokens[1]
        dur_2 = value_tokens[2]
        dur_3 = value_tokens[3]
        unit_1 = value_tokens[5]
        unit_2 = value_tokens[6]
        dur_4 = value_tokens[4]

        decoded_val = dur_ind
        decoded_val += dur_1
        if dur_2 != "[PAD]":
            decoded_val += dur_2
        if dur_3 != "[PAD]":
            decoded_val += dur_3
        if unit_1 != "[PAD]":
            decoded_val += unit_1
        if dur_4 != "[PAD]" and unit_2 != "[PAD]":
            decoded_val += dur_4 + unit_2
        return decoded_val


class D3(HeideltimeValue):
    regex = (
        "UNDEF-(this|next|last|REF|REFUNIT|REFDATE)?-?"
        "(day|month|year|decade|century'?|week|weekend|"
        "quarter|hour|minute|second|SU|WI|FA|SP|AU)?-??"
        "(monday|tuesday|wednesday|thursday|friday|"
        "saturday|sunday|january|february|march|april|"
        "may|june|july|august|september|october|november|"
        "december|WE|H1|H2|Q1|Q2|Q3|Q4|SU|WI|AU|FA|SP|XX|\d\d?)?"
        "(?:-?(\d\d?|XX))?"
        "(?:-(PLUS|MINUS|LESS)-(\d\d?)-?(\d\d?)?-?(\d\d?)?)?\)?"
        "(?:T(\d\d?|X|MO|MD|EV|AF|NI|MI|DT|XX)?(?::(\d\d?|XX))?(?:(?::|-)(\d\d|XX))?)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_this_next_last = m.group(1)
            f_unit = (
                m.group(2).replace("century'", "century") if m.group(2) else m.group(2)
            )
            f_desc = m.group(3)
            f_day = m.group(4)
            f_plus_minus = (
                m.group(5).replace("LESS", "MINUS") if m.group(5) else m.group(5)
            )
            f_plus_desc_1 = m.group(6)
            f_plus_desc_2 = m.group(7)
            f_plus_desc_3 = m.group(8)
            f_time_1 = m.group(9)
            f_time_2 = m.group(10)
            f_time_3 = m.group(11)

            return {
                "Basic": f_plus_minus,
                "Date_1": f_this_next_last,
                "Date_2": f_unit,
                "Date_3": f_desc,
                "Date_4": f_day,
                "Time_1": f_time_1,
                "Time_2": f_time_2,
                "Time_3": f_time_3,
                "Add_1": f_plus_desc_1,
                "Add_2": f_plus_desc_2,
                "Add_3": f_plus_desc_3,
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):
        this_next_last = value_tokens[1]
        unit = value_tokens[2]
        desc = value_tokens[3]
        day = value_tokens[4]

        plus_minus = value_tokens[0]
        plus_desc_1 = value_tokens[8]
        plus_desc_2 = value_tokens[9]
        plus_desc_3 = value_tokens[10]

        time_1 = value_tokens[5]
        time_2 = value_tokens[6]
        time_3 = value_tokens[7]

        if unit in ["1", "01"]:
            unit = "january"
            desc = "[PAD]"
        elif unit in ["2", "02"]:
            unit = "february"
            desc = "[PAD]"
        elif unit in ["3", "03"]:
            unit = "march"
            desc = "[PAD]"
        elif unit in ["4", "04"]:
            unit = "april"
            desc = "[PAD]"
        elif unit in ["5", "05"]:
            unit = "may"
            desc = "[PAD]"
        elif unit in ["6", "06"]:
            unit = "june"
            desc = "[PAD]"
        elif unit in ["7", "07"]:
            unit = "july"
            desc = "[PAD]"
        elif desc in ["8", "08"]:
            unit = "august"
            desc = "[PAD]"
        elif unit in ["9", "09"]:
            unit = "september"
            desc = "[PAD]"
        elif unit in ["10"]:
            unit = "october"
            desc = "[PAD]"
        elif unit in ["11"]:
            unit = "november"
            desc = "[PAD]"
        elif unit in ["12"]:
            unit = "december"
            desc = "[PAD]"

        decoded_val = "UNDEF-"
        if this_next_last != "[PAD]":
            decoded_val += this_next_last
        if unit != "[PAD]":
            if not decoded_val.endswith("-"):
                decoded_val += "-"
            decoded_val += unit
        if desc != "[PAD]":
            if not decoded_val.endswith("-"):
                decoded_val += "-"
            decoded_val += desc
        if day != "[PAD]":
            if not decoded_val.endswith("-"):
                decoded_val += "-"
            decoded_val += day

        if plus_minus != "[PAD]":
            decoded_val += "-" + plus_minus
            if plus_desc_1 != "[PAD]":
                decoded_val += "-" + plus_desc_1
                if plus_desc_2 != "[PAD]":
                    decoded_val += plus_desc_2
                    if plus_desc_3 != "[PAD]":
                        decoded_val += plus_desc_3

        if time_1 != "[PAD]":
            decoded_val += "T" + time_1
            if time_2 != "[PAD]":
                decoded_val += ":" + time_2
                if time_3 != "[PAD]":
                    decoded_val += ":" + time_3

        return decoded_val


class D4(HeideltimeValue):
    regex = (
        "UNDEF-(year|decade|century'?)-?"
        "(\d\d?|X)?-?"
        "(\d\d?|X)?-?"
        "(\d\d?|X|M|H|Q|SU|WI|AU|FA|SP|H1|H2|Q1|Q2|Q3|Q4|M1|M2)?\)?"
        "(?:T(\d\d?|X|MO|MD|EV|AF|NI|MI|DT)?(?::(\d\d?|XX))?(?:(?::|-)(\d\d|XX))?)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_unit = m.group(1).replace("century'", "century")
            f_num_season = m.group(2)
            f_num_2 = m.group(3)
            f_num_3 = m.group(4)
            f_time_1 = m.group(5)
            f_time_2 = m.group(6)
            f_time_3 = m.group(7)

            return {
                "Basic": "[PAD]",
                "Date_1": f_unit,
                "Date_2": f_num_season,
                "Date_3": f_num_2,
                "Date_4": f_num_3,
                "Time_1": f_time_1,
                "Time_2": f_time_2,
                "Time_3": f_time_3,
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):

        if assertions:
            assert value_tokens[0] == "[PAD]"
            assert value_tokens[8] == "[PAD]"
            assert value_tokens[9] == "[PAD]"
            assert value_tokens[10] == "[PAD]"

        unit = value_tokens[1]
        num_1 = value_tokens[2]
        num_2 = value_tokens[3]
        num_3 = value_tokens[4]
        time_1 = value_tokens[5]
        time_2 = value_tokens[6]
        time_3 = value_tokens[7]

        decoded_val = "UNDEF-"
        decoded_val += unit

        if num_1 != "[PAD]":
            if not decoded_val.endswith("-") and not decoded_val.endswith("century"):
                decoded_val += "-"
            decoded_val += num_1
        if num_2 != "[PAD]" and num_2 not in ["year", "day"]:
            if not decoded_val.endswith("-"):
                decoded_val += "-"
            decoded_val += num_2
        if num_3 != "[PAD]":
            if not decoded_val.endswith("-"):
                decoded_val += "-"
            decoded_val += num_3

        if time_1 != "[PAD]":
            decoded_val += "T" + time_1
            if time_2 != "[PAD]":
                decoded_val += ":" + time_2
                if time_3 != "[PAD]":
                    decoded_val += ":" + time_3
        return decoded_val


class D5(HeideltimeValue):
    regex = (
        "(UNDEF-year|UNDEF-this-year|UNDEF-century'?\d\d|\d\d\d\d)-"
        "(\d\d)-00 "
        "funcDateCalc\("
        "(WeekdayRelativeTo|EasterSundayOrthodox|EasterSunday|ShroveTideOrthodox)"
        "\(YEAR(?:(?:-(\d\d)))?(?:-(\d\d))?"
        "(?:,\s?(-?\d\d?))?(?:,\s?(-?\d\d?))?(?:, (true|false))?\)\)?"
    )

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_start = m.group(1)
            if f_start == "UNDEF-year":
                f_date = None
                f_year = "year"
            elif f_start == "UNDEF-this-year":
                f_date = "year"
                f_year = "this"
            elif f_start.startswith("UNDEF-century"):
                f_date = f_start.replace("UNDEF-century'", "").replace(
                    "UNDEF-century", ""
                )
                f_year = "century"
            else:
                f_date = f_start[2:4]
                f_year = f_start[0:2]
            f_month = m.group(2)
            f_holiday = m.group(3)
            f_anchor_month = m.group(4)
            f_anchor_day = m.group(5)
            f_arg_1 = m.group(6)
            f_arg_2 = m.group(7)
            f_arg_3 = m.group(8)

            return {
                "Basic": f_holiday,
                "Date_1": f_year,
                "Date_2": f_date,
                "Date_3": f_anchor_month,
                "Date_4": f_anchor_day,
                "Time_1": f_month,
                "Time_2": "[PAD]",
                "Time_3": "[PAD]",
                "Add_1": f_arg_1,
                "Add_2": f_arg_2,
                "Add_3": f_arg_3,
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):

        if assertions:
            assert value_tokens[6] == "[PAD]"
            assert value_tokens[7] == "[PAD]"

        holiday = value_tokens[0]
        year = value_tokens[1]
        date = value_tokens[2]
        anchor_month = value_tokens[3]
        anchor_day = value_tokens[4]
        month = value_tokens[5]
        arg_1 = value_tokens[8]
        arg_2 = value_tokens[9]

        if holiday == "WeekdayRelativeTo/true":
            holiday = "WeekdayRelativeTo"
            arg_3 = "true"
        elif holiday == "WeekdayRelativeTo/false":
            holiday = "WeekdayRelativeTo"
            arg_3 = "false"
        else:
            arg_3 = value_tokens[10]

        if year == "year":
            decoded_val = "UNDEF-year"
        elif year == "this":
            decoded_val = "UNDEF-this-year"
        elif year == "century":
            decoded_val = "UNDEF-century" + date
        else:
            decoded_val = year + date
        decoded_val += "-" + month + "-00"
        decoded_val += " funcDateCalc(" + holiday + "("
        decoded_val += "YEAR"

        if anchor_month != "[PAD]":
            decoded_val += "-" + anchor_month
            if anchor_day != "[PAD]":
                decoded_val += "-" + anchor_day

        if arg_1 != "[PAD]":
            decoded_val += ", " + arg_1
        if arg_2 != "[PAD]":
            decoded_val += ", " + arg_2
        if arg_3 != "[PAD]":
            decoded_val += ", " + arg_3

        decoded_val += "))"
        return decoded_val


class D6(HeideltimeValue):
    regex = "(UNDEF-year)-(00)-00 (decadeCalc)\((-?\d\d?)\)?()()()()"

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_date = None
            f_year = "year"
            f_month = m.group(2)
            f_holiday = m.group(3)
            f_anchor_month = m.group(4)

            return {
                "Basic": f_holiday,
                "Date_1": f_year,
                "Date_2": f_date,
                "Date_3": f_anchor_month,
                "Date_4": "[PAD]",
                "Time_1": f_month,
                "Time_2": "[PAD]",
                "Time_3": "[PAD]",
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):

        if assertions:
            assert value_tokens[0] == "decadeCalc"
            assert value_tokens[4] == "[PAD]"
            assert value_tokens[6] == "[PAD]"
            assert value_tokens[7] == "[PAD]"
            assert value_tokens[8] == "[PAD]"
            assert value_tokens[9] == "[PAD]"
            assert value_tokens[10] == "[PAD]"

        month = value_tokens[5]
        anchor_month = value_tokens[3]

        decoded_val = "UNDEF-year-" + month + "-00"
        decoded_val += " decadeCalc(" + anchor_month + ")"
        return decoded_val


class T1(HeideltimeValue):
    regex = "(DATE|TIME|DURATION|SET)"

    @classmethod
    def parseString(cls, value: str):
        m = re.match("^" + cls.regex + "$", value)
        if m:
            f_type = m.group(1)

            return {
                "Basic": f_type,
                "Date_1": "[PAD]",
                "Date_2": "[PAD]",
                "Date_3": "[PAD]",
                "Date_4": "[PAD]",
                "Time_1": "[PAD]",
                "Time_2": "[PAD]",
                "Time_3": "[PAD]",
                "Add_1": "[PAD]",
                "Add_2": "[PAD]",
                "Add_3": "[PAD]",
            }
        else:
            raise ValueError("Cannot parse")

    @classmethod
    def reconstructValue(cls, value_tokens: list[str], assertions: bool = False):
        raise ValueError()


class TimexTokenizer:
    def __init__(
        self,
        tokenizer,
        value_level_steps: int = 10_000,
        group_idx_special: int = 0,
        group_idx_text: int = 1,
        group_idx_markup: int = 2,
        group_idx_type: int = 3,
        group_idx_value: int = 4,
        group_idx_annotated: int = 5,
        mask_pct_text: float = 0.049,
        mask_pct_markup: float = 0.001,
        mask_pct_type: float = 0.10,
        mask_pct_value: float = 0.70,
        mask_pct_annotated: float = 0.15,
        ignore_case: bool = False,
        bert_like: bool = False,
    ):
        self.value_level = 1
        self.value_level_counter = 0
        self.value_level_steps = value_level_steps
        self.tokenizer = tokenizer
        self.ignore_case = ignore_case
        self.bert_like = bert_like

        self.our_vocabulary_offset = len(self.tokenizer.get_vocab())
        self.timex_start = '<TIMEX3 type="'
        self.timex_mid = '" value="'
        self.timex_end = '">'
        self.timex_closing = "</TIMEX3>"

        self.fixed_vocab = [
            "[PAD]",
            "DATE",
            "TIME",
            "DURATION",
            "SET",
            self.timex_start,
            self.timex_mid,
            self.timex_end,
            self.timex_closing,
            "PRESENT",
            "PAST",
            "FUTURE",
            "_",
            "REF",
            "REFUNIT",
            "REFDATE",
            "PLUS",
            "MINUS",
            "DCT",
            "BC",
            "X",
            "XX",  #'XXXX',
            "MO",
            "MD",
            "EV",
            "AF",
            "NI",
            "MI",
            "DT",
            "SP",
            "SU",
            "FA",
            "AU",
            "WI",
            "W",
            "WE",
            "H1",
            "H2",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "M1",
            "M2",
            "P",
            "PT",
            "T",
            "H",
            "D",
            "DE",
            "M",
            "C",
            "Y",
            "C",
            "CE",
            "W",
            "WE",
            "Qu",
            "Q",
            "S",
            "UNDEF",
            "-",
            "this",
            "next",
            "last",
            "UNIT",
            "DATE",
            "YEAR",
            "day",
            "month",
            "year",
            "decade",
            "century",
            "week",
            "weekend",
            "quarter",
            "hour",
            "minute",
            "second",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
            "funcDateCalc",
            "decadeCalc",
            "(",
            ")",
            ",",
            "WeekdayRelativeTo",
            "EasterSundayOrthodox",
            "EasterSunday",
            "ShroveTideOrthodox",
            "WeekdayRelativeTo/true",
            "WeekdayRelativeTo/false",
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "33",
            "34",
            "35",
            "36",
            "37",
            "38",
            "39",
            "40",
            "41",
            "42",
            "43",
            "44",
            "45",
            "46",
            "47",
            "48",
            "49",
            "50",
            "51",
            "52",
            "53",
            "54",
            "55",
            "56",
            "57",
            "58",
            "59",
            "60",
            "61",
            "62",
            "63",
            "64",
            "65",
            "66",
            "67",
            "68",
            "69",
            "70",
            "71",
            "72",
            "73",
            "74",
            "75",
            "76",
            "77",
            "78",
            "79",
            "80",
            "81",
            "82",
            "83",
            "84",
            "85",
            "86",
            "87",
            "88",
            "89",
            "90",
            "91",
            "92",
            "93",
            "94",
            "95",
            "96",
            "97",
            "98",
            "99",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "-1",
            "-2",
            "-3",
            "-4",
            "-5",
            "-6",
            "-7",
            "-8",
            "-9",
            "-10",
            "-11",
            "-12",
            "-13",
            "-14",
            "-15",
            "-16",
            "-17",
            "-18",
            "-19",
            "-20",
            "-21",
            "-22",
            "-23",
            "-24",
            "-25",
            "-26",
            "-27",
            "-28",
            "-29",
            "-30",
            "-31",
            "-32",
            "-33",
            "-34",
            "-35",
            "-36",
            "-37",
            "-38",
            "-39",
            "-40",
            "-41",
            "-42",
            "-43",
            "-44",
            "-45",
            "-46",
            "-47",
            "-48",
            "-49",
            "-50",
            "-51",
            "-52",
            "-53",
            "-54",
            "-55",
            "-56",
            "-57",
            "-58",
            "-59",
            "-60",
            "-61",
            "-62",
            "-63",
            "-64",
            "-65",
            "-66",
            "-67",
            "-68",
            "-69",
            "-70",
            "-71",
            "-72",
            "-73",
            "-74",
            "-75",
            "-76",
            "-77",
            "-78",
            "-79",
            "-80",
            "-81",
            "-82",
            "-83",
            "-84",
            "-85",
            "-86",
            "-87",
            "-88",
            "-89",
            "-90",
            "-91",
            "-92",
            "-93",
            "-94",
            "-95",
            "-96",
            "-97",
            "-98",
            "-99",
            "true",
            "false",
            ".",
        ]
        self.fixed_vocab_to_ids = {
            m: idx + self.our_vocabulary_offset
            for idx, m in enumerate(self.fixed_vocab)
        }
        self.fixed_vocab_ids_to_text = {
            idx: m for m, idx in self.fixed_vocab_to_ids.items()
        }

        self.group_idx_special = group_idx_special
        self.group_idx_text = group_idx_text
        self.group_idx_markup = group_idx_markup
        self.group_idx_type = group_idx_type
        self.group_idx_value = group_idx_value
        self.group_idx_annotated = group_idx_annotated
        self.mask_pct_text = mask_pct_text
        self.mask_pct_markup = mask_pct_markup
        self.mask_pct_type = mask_pct_type
        self.mask_pct_value = mask_pct_value
        self.mask_pct_annotated = mask_pct_annotated
        _all = [
            mask_pct_text,
            mask_pct_markup,
            mask_pct_type,
            mask_pct_value,
            mask_pct_annotated,
        ]
        assert sum(_all) == 1.0, "Does not count up: " + str(sum(_all))

    @staticmethod
    def _get_index_for_token(
        token: str, text: str, offset: int, max_len: int = 20
    ) -> Tuple[int, int]:
        """Get the text offset for a single token"""
        token_len = len(token)
        for search_offset in range(offset, len(text)):
            if text[search_offset : search_offset + token_len] == token:
                return search_offset, search_offset + token_len
            if search_offset > offset + max_len:
                return -1, -1
        return -2, -2

    def get_offsets(
        self,
        text: str,
        subwords: List[str],
        offset: int = 0,
        max_len: int = 20,
    ) -> List[Tuple[int, int]]:
        """Get the text offset for a list of tokens"""
        offsets = []
        start = offset
        if self.ignore_case:
            text = text.lower()
        for token in subwords:
            if token[0] == "▁":
                token = token[1:]
            elif token.startswith("##"):
                token = token[2:]
            if self.ignore_case:
                token = token.lower()
            end_limit = start + len(token) + max_len
            s, e = TimexTokenizer._get_index_for_token(token, text, start, end_limit)
            start = max(e if e >= 0 else start - abs(e), 0)
            offsets.append((s, e))
        return offsets

    def text_tokenization(self, text: str) -> tuple[list, list, list]:
        enc = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(enc)
        if len(tokens) == 0 or len(tokens[1:-1]) == 0:
            return [], [], []
        enc = enc[1:-1]
        tokens = tokens[1:-1]  # strip special tokens
        offsets = self.get_offsets(text, tokens, 0)
        return enc, tokens, offsets

    def value_tokenization(
        self, value: str, only_type: bool = False
    ) -> tuple[list, list]:
        val_cs = [T1, D1, D2, D4, D3, D5, D6, P1]
        d = {}
        for classifier in val_cs:
            try:
                d = classifier.parseString(value)
                break
            except:
                pass

        fields = [
            "Basic",
            "Date_1",
            "Date_2",
            "Date_3",
            "Date_4",
            "Time_1",
            "Time_2",
            "Time_3",
            "Add_1",
            "Add_2",
            "Add_3",
        ]
        if only_type:
            fields = ["Basic"]

        tokens = []
        for field in fields:
            x = d[field]
            x = "[PAD]" if x is None else x
            x = "[PAD]" if not x.strip() else x
            tokens.append(x)

        enc = [self.fixed_vocab_to_ids[x] for x in tokens]
        return enc, tokens

    def get_mask_group(self) -> int:
        r = random.random()

        t = self.mask_pct_text
        if r <= t:
            return self.group_idx_text

        t += self.mask_pct_markup
        if r <= t:
            return self.group_idx_markup

        t += self.mask_pct_type
        if r <= t:
            return self.group_idx_type

        t += self.mask_pct_value
        if r <= t:
            return self.group_idx_value

        t += self.mask_pct_annotated
        if r <= t:
            return self.group_idx_annotated
        raise ValueError("This should not happen: " + str(r))

    def compute_mask(
        self,
        groups: list[int],
        value_masking_method: str = "increasing",
        value_masking_level: int = -1,
        value_max_len: int = 11,
        markup_masking_level: int = 1,
        text_masking_level: int = 2,
        ann_masking_level: int = 2,
        logger=None,
    ) -> list[int]:

        if value_masking_method == "increasing":
            if value_masking_level < 0:
                value_masking_level = value_max_len

            self.value_level_counter += 1
            if self.value_level_counter % self.value_level_steps == 0:
                prev_level = self.value_level
                self.value_level = min(self.value_level + 1, 11)
                if logger is not None and self.value_level != prev_level:
                    logger.info(f"Increasing value masking level to {self.value_level}")
            value_masking_level = self.value_level
        elif value_masking_method == "random":
            r = random.randint(1, value_max_len * 2)
            value_masking_level = min(r, value_max_len)
        elif value_masking_method == "full":
            value_masking_level = value_masking_level

        value_ids = [x for x in range(value_max_len)]
        markup_ids = [x for x in range(4)]
        text_ids = [
            x for x in range(sum([1 for g in groups if g == self.group_idx_text]))
        ]
        annotated_ids = [
            x for x in range(sum([1 for g in groups if g == self.group_idx_annotated]))
        ]

        markup_masking_level = min(markup_masking_level, len(markup_ids))
        text_masking_level = min(text_masking_level, len(text_ids))
        ann_masking_level = min(ann_masking_level, len(annotated_ids))

        group_to_mask = self.get_mask_group()

        value_pos = -1
        markup_pos = -1
        text_pos = -1
        ann_pos = -1

        masks = []
        for g in groups:

            if g == self.group_idx_value and group_to_mask == self.group_idx_value:
                if value_pos < 0 or value_pos >= value_max_len:
                    value_pos = 0
                    sampled_value_ids = random.sample(value_ids, value_masking_level)
                masks.append(1 if value_pos in sampled_value_ids else 0)
                value_pos += 1

            elif g == self.group_idx_type and group_to_mask == self.group_idx_type:
                masks.append(1)

            elif g == self.group_idx_markup and group_to_mask == self.group_idx_markup:
                if markup_pos < 0 or markup_pos >= len(markup_ids):
                    markup_pos = 0
                    sampled_markup_ids = random.sample(markup_ids, markup_masking_level)
                masks.append(1 if markup_pos in sampled_markup_ids else 0)
                markup_pos += 1

            elif (
                g == self.group_idx_annotated
                and group_to_mask == self.group_idx_annotated
            ):
                if ann_pos < 0 or ann_pos >= len(annotated_ids):
                    ann_pos = 0
                    sampled_ann_ids = random.sample(annotated_ids, ann_masking_level)
                masks.append(1 if ann_pos in sampled_ann_ids else 0)
                ann_pos += 1

            elif g == self.group_idx_text and group_to_mask == self.group_idx_text:
                if text_pos < 0 or text_pos >= len(text_ids):
                    text_pos = 0
                    sampled_text_ids = random.sample(text_ids, text_masking_level)
                masks.append(1 if text_pos in sampled_text_ids else 0)
                text_pos += 1

            else:
                masks.append(0)
        return masks

    def tokenize_sentence(
        self,
        sent: dict[str, any],
        doc: dict[str, any] = None,
        use_cir_value: bool = True,
        use_dct: bool = False,
    ) -> tuple[list, list, list, str]:
        sent_text = sent["text"]  # .strip()
        last_end = 0

        tokens, enc, groups = [], [], []

        if use_dct:
            sent_tagged = f"<DCT>{doc['meta_data']['dct']}</DCT> "
            dct_enc, dct_tokens = self.value_tokenization(doc["meta_data"]["dct"])
            enc.extend(dct_enc)
            tokens.extend(dct_tokens)
            groups.extend([self.group_idx_text for _ in dct_tokens])
        else:
            sent_tagged = ""

        for annotation in sent["timex3"]:
            start = annotation["start"]
            end = annotation["end"]
            timex_type = annotation["type"]
            timex_value = annotation["cir_value" if use_cir_value else "value"]

            # Tokenize previous texts
            tmp_enc, tmp_tokens, _ = self.text_tokenization(sent_text[last_end:start])
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.group_idx_text for _ in tmp_tokens])

            # Tokenize annotation start
            enc.append(self.fixed_vocab_to_ids[self.timex_start])
            tokens.append(self.timex_start)
            groups.append(self.group_idx_markup)

            # Tokenize type
            tmp_enc, tmp_tokens = self.value_tokenization(timex_type, only_type=True)
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.group_idx_type for _ in tmp_tokens])

            # Tokenize annotation mid
            enc.append(self.fixed_vocab_to_ids[self.timex_mid])
            tokens.append(self.timex_mid)
            groups.append(self.group_idx_markup)

            # Tokenize value
            try:
                tmp_enc, tmp_tokens = self.value_tokenization(timex_value)
            except KeyError as e:
                print("Cannot process " + timex_value)
                raise e

            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.group_idx_value for _ in tmp_tokens])

            # Tokenize annotation end
            enc.append(self.fixed_vocab_to_ids[self.timex_end])
            tokens.append(self.timex_end)
            groups.append(self.group_idx_markup)

            # Tokenize annotated text
            tmp_enc, tmp_tokens, _ = self.text_tokenization(sent_text[start:end])
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.group_idx_annotated for _ in tmp_tokens])

            # Tokenize annotation closing
            enc.append(self.fixed_vocab_to_ids[self.timex_closing])
            tokens.append(self.timex_closing)
            groups.append(self.group_idx_markup)

            prev_text = sent_text[last_end:start]
            a1 = f'<TIMEX3 type="{timex_type}" value="{timex_value}">'
            a2 = sent_text[start:end]
            a3 = f"</TIMEX3>"
            sent_tagged += prev_text + a1 + a2 + a3
            last_end = end

        if last_end > 0:
            sent_tagged += sent_text[last_end:]
            tmp_enc, tmp_tokens, _ = self.text_tokenization(sent_text[last_end:])
            enc.extend(tmp_enc)
            tokens.extend(tmp_tokens)
            groups.extend([self.group_idx_text for _ in tmp_tokens])
        else:
            sent_tagged = sent_text
            enc, tokens, _ = self.text_tokenization(sent_text)
            groups = [self.group_idx_text for _ in tokens]

        if self.bert_like:
            enc = [self.tokenizer.cls_token_id] + enc + [self.tokenizer.sep_token_id]
            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        else:
            enc = [self.tokenizer.bos_token_id] + enc + [self.tokenizer.eos_token_id]
            tokens = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
        groups = [self.group_idx_special] + groups + [self.group_idx_special]

        return enc, tokens, groups, sent_tagged

    def decode_element(self, idx: int) -> str:
        if idx in self.fixed_vocab_ids_to_text:
            return self.fixed_vocab_ids_to_text[idx]
        else:
            return self.tokenizer.decode(idx)

    def decode_text_ids(self, input_ids: list[int]) -> list[str]:
        decoded = []
        for idx in input_ids:
            decoded.append(self.decode_element(idx))
        return decoded

    def decode_value_ids(self, value_ids: list[int]) -> list[str]:
        value_tokens = [self.fixed_vocab_ids_to_text[idx] for idx in value_ids]
        return self.decode_value_tokens(value_tokens)

    @staticmethod
    def decode_value_tokens(value_tokens: list[str]) -> str:
        if value_tokens[0] in ["PRESENT", "PAST", "FUTURE"]:
            decoded_val = D1.reconstructValue(value_tokens)

        elif value_tokens[0] in ["P", "PT"]:
            decoded_val = P1.reconstructValue(value_tokens)

        elif value_tokens[0] in ["[PAD]", "BC"] and re.match(
            "(\d\d?|XX)", value_tokens[1]
        ):
            decoded_val = D2.reconstructValue(value_tokens)

        elif value_tokens[0] in ["[PAD]", "PLUS", "MINUS"] and value_tokens[1] in [
            "this",
            "next",
            "last",
            "REF",
            "REFUNIT",
            "REFDATE",
        ]:
            decoded_val = D3.reconstructValue(value_tokens)

        elif value_tokens[0] == "[PAD]" and value_tokens[1] in [
            "year",
            "decade",
            "century",
        ]:
            decoded_val = D4.reconstructValue(value_tokens)

        elif value_tokens[0] in [
            "WeekdayRelativeTo",
            "WeekdayRelativeTo/false",
            "WeekdayRelativeTo/true",
            "EasterSundayOrthodox",
            "EasterSunday",
            "ShroveTideOrthodox",
        ]:
            decoded_val = D5.reconstructValue(value_tokens)

        elif value_tokens[0] == "decadeCalc":
            decoded_val = D6.reconstructValue(value_tokens)

        else:
            decoded_val = D3.reconstructValue(value_tokens)

        return decoded_val
