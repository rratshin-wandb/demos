from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
import psycopg2


class Printer:
    """
    Simple wrapper to stream status updates. Used by the financial bot
    manager as it orchestrates planning, search and writing.
    """

    def editDbP(self, s_query, a_values):
        i_return = -1
        try:
            conn = psycopg2.connect(database="xxxx",
            host="xxxx",
            user="xxxx",
            password="xxxx",
            port="5432")
            cursor = conn.cursor()

            # postgres_insert_query = """ INSERT INTO mobile (ID, MODEL, PRICE) VALUES (%s,%s,%s)"""
            # record_to_insert = (5, 'One Plus 6', 950)
            # cursor.execute(postgres_insert_query, record_to_insert)
            cursor.execute(s_query, a_values)

            conn.commit()
            count = cursor.rowcount
            print(count, "Record inserted successfully into table")

            i_return = cursor.fetchone()[0]

        except (Exception, psycopg2.Error) as error:
            print("Failed to insert record into table", error)
        # i_return = -2

        finally:
            # closing database connection.
            if conn:
                cursor.close()
                conn.close()
                print("PostgreSQL connection is closed")

        return i_return

    def __init__(self, console: Console) -> None:
        self.live = Live(console=console)
        self.items: dict[str, tuple[str, bool]] = {}
        self.hide_done_ids: set[str] = set()
        self.live.start()
        self.analysisid = -1

    def end(self) -> None:
        self.live.stop()

    def hide_done_checkmark(self, item_id: str) -> None:
        self.hide_done_ids.add(item_id)

    def set_analysisid(self, analysisid: int) -> None:
        self.analysisid = analysisid

    def update_item(
        self, item_id: str, content: str, is_done: bool = False, hide_checkmark: bool = False
    ) -> None:
        
        if not content.startswith("View trace"):
            s_query = "update t_financial_analyses set progress = concat(progress, '<br><br>', %s) where analysisid = " + str(self.analysisid)
            a_values = ([content])
            self.editDbP(s_query, a_values)

        self.items[item_id] = (content, is_done)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self.flush()

    def mark_item_done(self, item_id: str) -> None:
        self.items[item_id] = (self.items[item_id][0], True)
        self.flush()

    def flush(self) -> None:
        renderables: list[Any] = []
        for item_id, (content, is_done) in self.items.items():
            if is_done:
                prefix = "✅ " if item_id not in self.hide_done_ids else ""
                renderables.append(prefix + content)
            else:
                renderables.append(Spinner("dots", text=content))
        self.live.update(Group(*renderables))
