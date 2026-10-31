package udfx.runtime;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.sql.api.java.UDF2;

import scala.collection.Iterator;
import scala.collection.Seq;

public final class FrequencyOrder implements UDF2<Seq<String>, Long, List<String>> {
    @Override
    public List<String> call(Seq<String> values, Long count) {
        if (values == null || count == null) {
            return null;
        }
        if (count <= 0) {
            return List.of();
        }
        Map<String, Integer> counts = new LinkedHashMap<>();
        Iterator<String> items = values.iterator();
        while (items.hasNext()) {
            counts.merge(items.next(), 1, Integer::sum);
        }
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort((left, right) -> Integer.compare(right.getValue(), left.getValue()));
        int length = (int) Math.min(count, entries.size());
        List<String> result = new ArrayList<>(length);
        for (int index = 0; index < length; index++) {
            result.add(entries.get(index).getKey());
        }
        return result;
    }
}
