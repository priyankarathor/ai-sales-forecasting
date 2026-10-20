from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class UserSalesforceSchema(BaseModel):
    user_id: str = Field(min_length=1)
    email_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    phone: str
    role: str
    department: str
    region: str
    status: str
    created_at: str = Field(min_length=1)
    last_login_at: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_excel_values(cls, value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        if isinstance(value, str):
            return value.strip()
        return value
