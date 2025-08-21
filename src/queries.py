
cleaning_query = lambda x, y: f"""
UPDATE demo_user.{x}
SET {y} = LOWER(
    TRIM(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE({y}, '[[:digit:]]+', ''), 
                    '[-_/\\|]', ' '
                ),
                '[[:punct:]]', ' '
            ),
            '\s+', ' '       -- collapse multiple spaces into one
        )
    )
);
"""