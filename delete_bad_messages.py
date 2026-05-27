import sqlite3
import re
conn = sqlite3.connect("/Users/zmll/.hermes/state.db")
c = conn.cursor()

# We need to wipe the broken partial runs so we don't trip over them on resume.
c.execute("DELETE FROM messages WHERE id IN (76408, 76407, 76406)")

conn.commit()
conn.close()
