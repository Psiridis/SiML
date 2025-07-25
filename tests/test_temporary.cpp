#include <gtest/gtest.h>
#include "temporary.hpp"

TEST(SiML_Addition, positive_addition) {
    EXPECT_EQ(SiML::addition(1, 1), 2);
}

TEST(SiML_Addition, negative_addition) {
    EXPECT_EQ(SiML::addition(-1, -1), -2);
}