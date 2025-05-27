import csv
import sys

def parse_line(line):
    """Parse line based on the last comma
    
    Parameters
    ----------
    line : str
        line from file
    
    Output
    ------
    line[0:last_comma] : str
        the line up to the last comma
    line[last_comma+1:len(line)] : str
        the line after the last comma
    """
    last_comma=line.rfind(",")
    return line[0:last_comma], line[last_comma+1:len(line)]

if __name__ == "__main__":
    line = sys.argv[1]
    try:
        drug_name, drug_id = parse_line(line)
        print(drug_name+"\n"+drug_id)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

