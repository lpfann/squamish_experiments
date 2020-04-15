import pandas as pd
import pathlib
import os

OUTPUT_PATH = pathlib.Path(__file__).parent / ("./output/tables/")


def print_df_astable(
    df, filename, output_root=OUTPUT_PATH, folder="", decimals=2, na_rep="NaN"
):
    """ Output Dataframe as Latex Table in .tex file.
    
    Arguments:
        df {pd.DataFrame} -- Pandas DataFrame to print
        filename {str} -- Name of output file without file ending (we add .tex )
    
    Keyword Arguments:
        output_root {str} -- Root folder (default: ./output/tables/)
        folder {str} -- Subfolder which is added to output_root (default: {""})
    
    Returns:
        {str} -- Latex Table Code
    """
    print("Saving table as Latex output")
    output = df.to_latex(
        multicolumn=True,
        multirow=True,
        bold_rows=True,
        float_format=f"%.{decimals}f",
        na_rep=na_rep,
    )
    print(output)
    output_path = output_root / folder
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "{}.tex".format(filename), "w") as f:
        f.write(output)
    return output
