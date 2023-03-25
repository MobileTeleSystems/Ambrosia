import typing as tp

import enum


class AmbrosiaEnum(enum.Enum):
    '''
    Custom enum for Ambrosia
    '''
    def _check_for_existing_members():
        '''
        To implement custom enum with inheretance
        with addtional methods 
        '''
        pass

    @classmethod
    def check_value_in_enum(cls, value: tp.Any) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def get_all_enum_values(cls) -> tp.List[str]:
        return [it.value for it in cls._member_map_.values()]

    @classmethod
    def raise_if_value_incorrect_enum(cls, value: tp.Any) -> None:
        if not cls.check_value_in_enum(value):
            msg: str = f"Choose value from " + ", ".join(cls.get_all_enum_values())
            raise ValueError(msg)


class Alternatives(AmbrosiaEnum):
    ts: str = 'two-sided'
    less: str = 'less'
    gr: str = 'greater'


class Effects(AmbrosiaEnum):
    abs: str = "absolute"
    rel: str = "relative"
