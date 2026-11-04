package udfx.runtime;

final class JsonEncoding {
    private static void hexadecimal(StringBuilder output, int value) {
        output.append("\\u");
        for (int shift = 12; shift >= 0; shift -= 4) {
            output.append(Character.forDigit((value >> shift) & 15, 16));
        }
    }

    static void character(StringBuilder output, int value, boolean ascii) {
        switch (value) {
            case '"': output.append("\\\""); return;
            case '\\': output.append("\\\\"); return;
            case '\b': output.append("\\b"); return;
            case '\t': output.append("\\t"); return;
            case '\n': output.append("\\n"); return;
            case '\f': output.append("\\f"); return;
            case '\r': output.append("\\r"); return;
            default: break;
        }
        if (value < 32 || (ascii && value >= 127)) {
            if (value <= 0xffff) {
                hexadecimal(output, value);
            } else {
                hexadecimal(output, Character.highSurrogate(value));
                hexadecimal(output, Character.lowSurrogate(value));
            }
        } else {
            output.appendCodePoint(value);
        }
    }
}
