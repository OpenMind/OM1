---
title: Keep-out Zones
description: "Mark no-go areas on a map that the robot's autonomy avoids."
icon: ban
---

A keep-out zone is an area of a map you want the robot to stay out of — a loading bay, a spill, a section under maintenance. You draw the zones as polygons on a map, and live consumers of the map (for example [person-following](person-following.md)) avoid them. Zones are stored per map, so each space carries its own no-go areas.

Under the hood, a saved zone set is written next to the map as `<zone_name>.keepout.geojson`, and a publisher node watches that file and re-publishes it on `/keepout/zones` for anything that needs it — so an update takes effect without a restart.

## Saving a zone set

Zones are **GeoJSON polygons**. Save a named set for a map:

```bash
curl -X POST http://<robot>:5000/maps/keepout/save \
  -H 'Content-Type: application/json' \
  -d '{
    "map_name": "office",
    "zone_name": "loading_bay",
    "geojson": {
      "type": "FeatureCollection",
      "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon",
           "coordinates": [[[1.0, 1.0], [3.0, 1.0], [3.0, 2.5], [1.0, 2.5], [1.0, 1.0]]]},
         "properties": {"name": "loading_bay"}}
      ]
    }
  }'
```

## Managing zones

```bash
curl "http://<robot>:5000/maps/keepout/list?map_name=office"        # all zone sets on a map
curl "http://<robot>:5000/maps/keepout/get?map_name=office&zone_name=loading_bay"
curl -X POST http://<robot>:5000/maps/keepout/delete \
  -H 'Content-Type: application/json' -d '{"map_name": "office", "zone_name": "loading_bay"}'
```

## Parameters

`POST /maps/keepout/save`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `map_name` | string | yes | The map the zones belong to (must already exist) |
| `zone_name` | string | yes | Name for this zone set — no `/`, `\`, `..`, or spaces |
| `geojson` | object | yes | A GeoJSON `FeatureCollection` of `Polygon` features |

## If something goes wrong

- **`400` "required"** — `map_name`, `zone_name`, and `geojson` are all mandatory.
- **`400` "Invalid zone_name"** — the name contains `/`, `\`, `..`, or a space.
- **`400` map missing** — the map directory doesn't exist; save the [map](maps-routes-locations.md) first.

For system endpoints like `GET /status` and base control, see the [Autonomy API Overview](../api_endpoints.md).
