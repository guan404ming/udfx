package udfx.runtime;

import org.apache.spark.sql.api.java.UDF2;

import scala.collection.Iterator;
import scala.collection.Seq;

public final class JsonStrings implements UDF2<Seq<String>, Boolean, String> {
    @Override
    public String call(Seq<String> values, Boolean ascii) {
        if (ascii == null) {
            return null;
        }
        if (values == null) {
            return "null";
        }
        StringBuilder output = new StringBuilder("[");
        Iterator<String> items = values.iterator();
        boolean previous = false;
        while (items.hasNext()) {
            String value = items.next();
            if (previous) {
                output.append(", ");
            }
            previous = true;
            if (value == null) {
                output.append("null");
                continue;
            }
            output.append('"');
            for (int offset = 0; offset < value.length();) {
                int point = value.codePointAt(offset);
                offset += Character.charCount(point);
                JsonEncoding.character(output, point, ascii);
            }
            output.append('"');
        }
        output.append(']');
        return output.toString();
    }
}
