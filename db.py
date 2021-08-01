import sqlite3


db = sqlite3.connect('people.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
dbase = db.cursor()


def create_table():
    dbase.execute('CREATE TABLE IF NOT EXISTS people(total_entered REAL, total_left REAL, datestamp TEXT)')


def data_entry(totalEntered, totalLeft, datestamp):

    db = sqlite3.connect('people.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()
    sqlite_insert_with_param = """INSERT INTO 'people'
                          ('total_entered', 'total_left', 'datestamp') 
                          VALUES (?, ?, ?);"""
    data_tuple = (totalEntered, totalLeft, datestamp)
    dbase.execute(sqlite_insert_with_param, data_tuple)
    db.commit()



def read_from_db():
    
    db = sqlite3.connect('people.db', detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    dbase = db.cursor()

    dbase.execute("SELECT total_entered, total_left, datestamp FROM people")
    data = dbase.fetchall()

    totalEntered_list = []
    totalLeft_list = []
    timestamp_list = []

    for tuple in data:
        totalE = tuple[0]
        totalL = tuple[1]
        timestamp = tuple[2]

        totalEntered_list.append(totalE)
        totalLeft_list.append(totalL)
        timestamp_list.append(timestamp)

    return totalEntered_list, totalLeft_list, timestamp_list


create_table()
