class Reader:
  @staticmethod
  def csv(input_file_name):
    f = open(input_file_name, "r")
    full_page = f.read()
    rows = [
      [
        float(c) for c in rows.split(",")
      ] for rows in full_page.split("\n") if rows != ''
    ]

    return rows
