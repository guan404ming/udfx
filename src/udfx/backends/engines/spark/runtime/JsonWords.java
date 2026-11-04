package udfx.runtime;

import org.apache.spark.sql.api.java.UDF3;

public final class JsonWords implements UDF3<String, Boolean, String, String> {
    @Override
    public String call(String value, Boolean ascii, String whitespace) {
        if (value == null || ascii == null || whitespace == null) {
            return null;
        }
        long lowerSpaces = 0;
        long upperSpaces = 0;
        for (int offset = 0; offset < whitespace.length();) {
            int point = whitespace.codePointAt(offset);
            offset += Character.charCount(point);
            if (point < 64) {
                lowerSpaces |= 1L << point;
            } else if (point < 128) {
                upperSpaces |= 1L << point;
            }
        }
        StringBuilder output = new StringBuilder(value.length() + 2);
        output.append('[');
        boolean word = false;
        boolean previous = false;
        for (int offset = 0; offset < value.length();) {
            int point = value.codePointAt(offset);
            offset += Character.charCount(point);
            boolean space = point < 128
                ? (((point < 64 ? lowerSpaces : upperSpaces) & (1L << point)) != 0)
                : whitespace.indexOf(point) >= 0;
            if (space) {
                if (word) {
                    output.append('"');
                    word = false;
                }
            } else {
                if (!word) {
                    if (previous) {
                        output.append(", ");
                    }
                    output.append('"');
                    word = true;
                    previous = true;
                }
                JsonEncoding.character(output, point, ascii);
            }
        }
        if (word) {
            output.append('"');
        }
        output.append(']');
        return output.toString();
    }
}
