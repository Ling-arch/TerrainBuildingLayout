#pragma once

#include <raylib.h>
#include "util.h"

namespace renderUtil{

    Color room_color_from_id(const size_t room_id, const size_t num_room);

    std::vector<Color> room2colors(const size_t num_room);

    static inline Color AspectToColor(float aspect)
    {
        float hue = aspect * 180.0f / PI; // rad → deg
        return ColorFromHSV(hue, 0.85f, 0.9f);
    }
}