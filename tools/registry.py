import json
import time

from jsonschema import Draft202012Validator, ValidationError


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


class Registry:
    def __init__(self):
        self.tools = {}

    def add(self, name, description, schema, function):
        if name in self.tools:
            raise ValueError(f"Duplicate tool: {name}")
        Draft202012Validator.check_schema(schema)
        self.tools[name] = (description, schema, function)

    def schemas(self):
        return [{"type": "function", "function": {"name": name, "description": desc,
                 "parameters": schema}} for name, (desc, schema, _) in self.tools.items()]

    def execute(self, name, arguments):
        started = time.perf_counter()
        try:
            if name not in self.tools:
                raise ValueError(f"Unknown tool {name}. Available: {', '.join(self.tools)}")
            _, schema, function = self.tools[name]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            Draft202012Validator(schema).validate(arguments)
            result = function(**arguments)
            # Strict JSON round trip prevents accidental NaN/provider-invalid output.
            result = json.loads(json.dumps(result, allow_nan=False))
            response = {"ok": True, "result": result}
        except (ValueError, TypeError, KeyError, ValidationError) as error:
            message = error.message if isinstance(error, ValidationError) else str(error)
            response = {"ok": False, "error": message[:600]}
        except Exception:
            response = {"ok": False, "error": "Tool failed internally. Try a simpler operation."}
        response["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return response
