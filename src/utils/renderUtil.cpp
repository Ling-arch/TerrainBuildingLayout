#include "renderUtil.h"

namespace renderUtil{

    Color room_color_from_id(size_t room_id, size_t num_room)
    {
        if (room_id >= num_room)
            return BLACK;
        float t = float(room_id) / float(std::max<size_t>(1, num_room));
        float h = t * 360.0f; // hue
        float s = 0.6f;
        float v = 0.9f;

        float c = v * s;
        float x = c * (1 - std::fabsf(std::fmod(h / 60.0f, 2) - 1));
        float m = v - c;

        float r = 0, g = 0, b = 0;
        if (h < 60)
        {
            r = c;
            g = x;
        }
        else if (h < 120)
        {
            r = x;
            g = c;
        }
        else if (h < 180)
        {
            g = c;
            b = x;
        }
        else if (h < 240)
        {
            g = x;
            b = c;
        }
        else if (h < 300)
        {
            r = x;
            b = c;
        }
        else
        {
            r = c;
            b = x;
        }
        return Color{(unsigned char)((r + m) * 255), (unsigned char)((g + m) * 255), (unsigned char)((b + m) * 255), 255};
    }

    std::vector<Color> room2colors(const size_t num_room){
        std::vector<Color> room2colors;
        for(size_t i = 0; i < num_room; i++){
            room2colors.push_back(room_color_from_id(i,num_room));
        }
        return room2colors;
    }
}