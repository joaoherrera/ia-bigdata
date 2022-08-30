import cx_Oracle
from matplotlib.pyplot import fill

def init_cx_oracle():
    try:
        cx_Oracle.init_oracle_client(lib_dir="/usr/lib/oracle/21/client64/lib")
        print("libclntsh.so loaded successfully.")

    except Exception as e:
        print("Error when loading libclntsh.so, is Oracle Instant Client installed?")
        print(str(e))


def connect(user: str = None, password: str = None):
    if not user:
        user = "M36596684865"
    
    if not password:
        password = "M36596684865"
    
    dsn = cx_Oracle.makedsn(host="orclgrad1.icmc.usp.br", port="1521", service_name="pdb_junior.icmc.usp.br")
    connection = cx_Oracle.connect(user=user, password=password, dsn=dsn)
    
    print(connection.version)
    return connection


def read_sql(sql_file_path: str) -> str:
    try:
        with open(sql_file_path, mode="r") as sql_file:
            return sql_file.read()
    
    except Exception:
        print(f"Cannot read file {sql_file_path}.")
        return


def create_schema(sql_file_path: str, cursorDDL: object):
    sql_instructions = read_sql(sql_file_path)
        
    if sql_instructions:
        sql_instructions = sql_instructions.split(";")
        
        for instruction in sql_instructions:
            try:
                if instruction == '': break
                cursorDDL.execute(instruction)
        
            except Exception as msg:
                print(f"SQL Error: {str(msg)} {instruction}")

        print("Schema created successfully.")


def fill_database(sql_file_path: str, cursorDML: object, connection: object):
    sql_instructions = read_sql(sql_file_path)
    
    if sql_instructions:
        sql_instructions = sql_instructions.split(";")
        
        for instruction in sql_instructions:
            try:
                cursorDML.execute(instruction)
            
            except Exception as msg:
                print(f"SQL Error: {str(msg)} {instruction}")

        print("Instructions executed successfully.")
        connection.commit()


def show_tables(cursorDML: object):
    cursorDML.execute("select * from user_tables")
    
    for data in cursorDML:
        print(data[0])

def main():
    init_cx_oracle()
    connection = connect()
    
    # Cursors send the instructions to the DB
    cursorDDL = connection.cursor()
    cursorDML = connection.cursor()
    
    # create_schema("data/EsquemaFutebol/Esquema_Futebol.sql", cursorDDL)
    # fill_database("data/EsquemaFutebol/Dados_Futebol.sql", cursorDML, connection)
    show_tables(cursorDML)
    
if __name__ == "__main__":
    main()