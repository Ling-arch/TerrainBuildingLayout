#pragma once

#include <raylib.h>
#include "util.h"

namespace renderUtil
{

    Color room_color_from_id(const size_t room_id, const size_t num_room);

    std::vector<Color> room2colors(const size_t num_room);

    static inline Color AspectToColor(float aspect)
    {
        float hue = aspect * 180.0f / PI; // rad → deg
        return ColorFromHSV(hue, 0.85f, 0.9f);
    }

    Color ColorFromHue(float h);
    Color ColorFromLowHue(float h);
     inline Color mixColor(const std::vector<Color> &colors,
                                                      const std::vector<float> &weights)
    {
        float r = 0, g = 0, b = 0, a = 0;

        for (size_t i = 0; i < colors.size(); ++i)
        {
            r += weights[i] * colors[i].r;
            g += weights[i] * colors[i].g;
            b += weights[i] * colors[i].b;
            a += weights[i] * colors[i].a;
        }

        return Color{
            (unsigned char)std::clamp(r, 0.f, 255.f),
            (unsigned char)std::clamp(g, 0.f, 255.f),
            (unsigned char)std::clamp(b, 0.f, 255.f),
            (unsigned char)std::clamp(a, 0.f, 255.f)};
    }
}