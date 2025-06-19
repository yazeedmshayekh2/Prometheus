from db_utils import DatabaseConnection
from auth_db import AuthDB

db = DatabaseConnection()
auth_db = AuthDB()

print(db.get_family_members('28140001175')['PDFLink'].iloc[-1])

