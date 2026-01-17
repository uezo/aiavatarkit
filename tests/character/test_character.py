from datetime import date, timedelta
import os
from uuid import uuid4
import pytest
import pytest_asyncio
from aiavatar.character import (
    Character,
    WeeklySchedule,
    DailySchedule,
    Diary,
    ActivityRangeResult,
    CharacterRepository,
    ActivityRepository,
    CharacterService,
)

AIAVATAR_DB_PORT = os.getenv("AIAVATAR_DB_PORT", "5432")
AIAVATAR_DB_USER = os.getenv("AIAVATAR_DB_USER", "postgres")
AIAVATAR_DB_PASSWORD = os.getenv("AIAVATAR_DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


class FixtureContext:
    def __init__(self, service: CharacterService):
        self.service = service
        self.created_character_ids: list[str] = []

    async def create_character(self, **kwargs) -> Character:
        if "name" not in kwargs:
            kwargs["name"] = f"test_character_{uuid4()}"
        if "prompt" not in kwargs:
            kwargs["prompt"] = "Test prompt"
        character = await self.service.character.create(**kwargs)
        self.created_character_ids.append(character.id)
        return character

    async def cleanup(self):
        for character_id in self.created_character_ids:
            # Delete related data first (foreign key order)
            await self.service.activity.delete_weekly_schedule(character_id=character_id)
            # Delete daily schedules and diaries for recent dates (extended range for batch tests)
            for i in range(-10, 2):
                target_date = date.today() + timedelta(days=i)
                await self.service.activity.delete_daily_schedule(
                    character_id=character_id, schedule_date=target_date
                )
                await self.service.activity.delete_diary(
                    character_id=character_id, diary_date=target_date
                )
            await self.service.character.delete(character_id=character_id)


@pytest_asyncio.fixture
async def ctx():
    svc = CharacterService(
        openai_api_key=OPENAI_API_KEY,
        openai_model=OPENAI_MODEL,
        openai_reasoning_effort="none",
        port=int(AIAVATAR_DB_PORT),
        user=AIAVATAR_DB_USER,
        password=AIAVATAR_DB_PASSWORD,
    )
    await svc.get_pool()
    test_ctx = FixtureContext(svc)
    yield test_ctx
    await test_ctx.cleanup()
    if svc._pool is not None:
        await svc._pool.close()


# CharacterRepository tests

@pytest.mark.asyncio
async def test_character_create(ctx):
    name = f"test_character_{uuid4()}"
    character = await ctx.create_character(
        name=name,
        prompt="Test character prompt",
        metadata={"key": "value"}
    )
    assert character.name == name
    assert character.prompt == "Test character prompt"
    assert character.metadata == {"key": "value"}
    assert character.id is not None


@pytest.mark.asyncio
async def test_character_get(ctx):
    name = f"test_character_{uuid4()}"
    created = await ctx.create_character(name=name)
    fetched = await ctx.service.character.get(character_id=created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == name


@pytest.mark.asyncio
async def test_character_get_not_found(ctx):
    fetched = await ctx.service.character.get(character_id="nonexistent_id")
    assert fetched is None


@pytest.mark.asyncio
async def test_character_update(ctx):
    character = await ctx.create_character(prompt="Original prompt")
    updated = await ctx.service.character.update(
        character_id=character.id,
        name="Updated name",
        prompt="Updated prompt",
        metadata={"updated": True}
    )
    assert updated is not None
    assert updated.name == "Updated name"
    assert updated.prompt == "Updated prompt"
    assert updated.metadata == {"updated": True}


@pytest.mark.asyncio
async def test_character_delete(ctx):
    character = await ctx.create_character()
    deleted = await ctx.service.character.delete(character_id=character.id)
    assert deleted is True
    fetched = await ctx.service.character.get(character_id=character.id)
    assert fetched is None
    # Remove from tracking since already deleted
    ctx.created_character_ids.remove(character.id)


# ActivityRepository - WeeklySchedule tests

@pytest.mark.asyncio
async def test_weekly_schedule_create(ctx):
    character = await ctx.create_character()
    schedule = await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="Weekly schedule content"
    )
    assert schedule.character_id == character.id
    assert schedule.content == "Weekly schedule content"


@pytest.mark.asyncio
async def test_weekly_schedule_get(ctx):
    character = await ctx.create_character()
    created = await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="Weekly schedule content"
    )
    fetched = await ctx.service.activity.get_weekly_schedule(character_id=character.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.content == "Weekly schedule content"


@pytest.mark.asyncio
async def test_weekly_schedule_update(ctx):
    character = await ctx.create_character()
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="Original content"
    )
    updated = await ctx.service.activity.update_weekly_schedule(
        character_id=character.id,
        content="Updated content"
    )
    assert updated is not None
    assert updated.content == "Updated content"


@pytest.mark.asyncio
async def test_weekly_schedule_delete(ctx):
    character = await ctx.create_character()
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="Weekly schedule content"
    )
    deleted = await ctx.service.activity.delete_weekly_schedule(character_id=character.id)
    assert deleted is True
    fetched = await ctx.service.activity.get_weekly_schedule(character_id=character.id)
    assert fetched is None


# ActivityRepository - DailySchedule tests

@pytest.mark.asyncio
async def test_daily_schedule_create(ctx):
    character = await ctx.create_character()
    schedule_date = date.today()
    schedule = await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date,
        content="Daily schedule content",
        content_context={"key": "value"}
    )
    assert schedule.character_id == character.id
    assert schedule.schedule_date == schedule_date
    assert schedule.content == "Daily schedule content"
    assert schedule.content_context == {"key": "value"}


@pytest.mark.asyncio
async def test_daily_schedule_get(ctx):
    character = await ctx.create_character()
    schedule_date = date.today()
    created = await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date,
        content="Daily schedule content"
    )
    fetched = await ctx.service.activity.get_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date
    )
    assert fetched is not None
    assert fetched.id == created.id


@pytest.mark.asyncio
async def test_daily_schedule_update(ctx):
    character = await ctx.create_character()
    schedule_date = date.today()
    await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date,
        content="Original content"
    )
    updated = await ctx.service.activity.update_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date,
        content="Updated content"
    )
    assert updated is not None
    assert updated.content == "Updated content"


@pytest.mark.asyncio
async def test_daily_schedule_delete(ctx):
    character = await ctx.create_character()
    schedule_date = date.today()
    await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date,
        content="Daily schedule content"
    )
    deleted = await ctx.service.activity.delete_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date
    )
    assert deleted is True
    fetched = await ctx.service.activity.get_daily_schedule(
        character_id=character.id,
        schedule_date=schedule_date
    )
    assert fetched is None


# ActivityRepository - Diary tests

@pytest.mark.asyncio
async def test_diary_create(ctx):
    character = await ctx.create_character()
    diary_date = date.today()
    diary = await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=diary_date,
        content="Diary content",
        content_context={"mood": "happy"}
    )
    assert diary.character_id == character.id
    assert diary.diary_date == diary_date
    assert diary.content == "Diary content"
    assert diary.content_context == {"mood": "happy"}


@pytest.mark.asyncio
async def test_diary_get(ctx):
    character = await ctx.create_character()
    diary_date = date.today()
    created = await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=diary_date,
        content="Diary content",
        content_context={}
    )
    fetched = await ctx.service.activity.get_diary(
        character_id=character.id,
        diary_date=diary_date
    )
    assert fetched is not None
    assert fetched.id == created.id


@pytest.mark.asyncio
async def test_diary_update(ctx):
    character = await ctx.create_character()
    diary_date = date.today()
    await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=diary_date,
        content="Original diary",
        content_context={}
    )
    updated = await ctx.service.activity.update_diary(
        character_id=character.id,
        diary_date=diary_date,
        content="Updated diary",
        content_context={"updated": "yes"}
    )
    assert updated is not None
    assert updated.content == "Updated diary"
    assert updated.content_context == {"updated": "yes"}


@pytest.mark.asyncio
async def test_diary_delete(ctx):
    character = await ctx.create_character()
    diary_date = date.today()
    await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=diary_date,
        content="Diary content",
        content_context={}
    )
    deleted = await ctx.service.activity.delete_diary(
        character_id=character.id,
        diary_date=diary_date
    )
    assert deleted is True
    fetched = await ctx.service.activity.get_diary(
        character_id=character.id,
        diary_date=diary_date
    )
    assert fetched is None


# CharacterService generation tests

@pytest.mark.asyncio
async def test_generate_weekly_schedule(ctx):
    character_prompt = "高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    content = await ctx.service.generate_weekly_schedule(character_prompt=character_prompt)
    assert content is not None
    assert len(content) > 0
    assert "|" in content  # Markdown table


@pytest.mark.asyncio
async def test_generate_daily_schedule(ctx):
    character_prompt = "高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    weekly_schedule = "| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    content, context = await ctx.service.generate_daily_schedule(
        schedule_date=date.today(),
        character_prompt=character_prompt,
        weekly_schedule_content=weekly_schedule
    )
    assert content is not None
    assert len(content) > 0
    assert "source" in context
    assert "reasoning" in context


@pytest.mark.asyncio
async def test_generate_diary(ctx):
    character_prompt = "高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    daily_schedule = "| 時間帯 | 活動 |\n|---|---|\n| 7:00-8:00 | 起床・朝食 |"
    content, context = await ctx.service.generate_diary(
        character_prompt=character_prompt,
        diary_date=date.today(),
        daily_schedule_content=daily_schedule
    )
    assert content is not None
    assert len(content) > 0
    assert "private_source" in context
    assert "public_source" in context
    assert "topics" in context


# CharacterService high-level operation tests

@pytest.mark.asyncio
async def test_initialize_character(ctx):
    name = f"test_character_{uuid4()}"
    character, weekly_schedule, daily_schedule = await ctx.service.initialize_character(
        name=name,
        character_prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。",
        metadata={"test": True}
    )
    ctx.created_character_ids.append(character.id)

    # Verify returned objects
    assert character is not None
    assert character.name == name
    assert character.metadata == {"test": True}

    assert weekly_schedule is not None
    assert weekly_schedule.character_id == character.id
    assert len(weekly_schedule.content) > 0

    assert daily_schedule is not None
    assert daily_schedule.character_id == character.id
    assert daily_schedule.schedule_date == date.today()

    # Verify data persisted in database
    fetched_character = await ctx.service.character.get(character_id=character.id)
    assert fetched_character is not None
    assert fetched_character.name == name
    assert fetched_character.prompt == "高校2年生の女子。部活は吹奏楽部でフルートを担当。"

    fetched_weekly = await ctx.service.activity.get_weekly_schedule(character_id=character.id)
    assert fetched_weekly is not None
    assert fetched_weekly.id == weekly_schedule.id
    assert fetched_weekly.content == weekly_schedule.content

    fetched_daily = await ctx.service.activity.get_daily_schedule(
        character_id=character.id,
        schedule_date=date.today()
    )
    assert fetched_daily is not None
    assert fetched_daily.id == daily_schedule.id
    assert fetched_daily.content == daily_schedule.content


@pytest.mark.asyncio
async def test_create_weekly_schedule_with_generation(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    schedule = await ctx.service.create_weekly_schedule_with_generation(
        character_id=character.id
    )
    assert schedule is not None
    assert schedule.character_id == character.id
    assert len(schedule.content) > 0


@pytest.mark.asyncio
async def test_create_daily_schedule_with_generation(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    )
    schedule = await ctx.service.create_daily_schedule_with_generation(
        character_id=character.id,
        schedule_date=date.today()
    )
    assert schedule is not None
    assert schedule.character_id == character.id
    assert schedule.schedule_date == date.today()


@pytest.mark.asyncio
async def test_create_diary_with_generation(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=date.today(),
        content="| 時間帯 | 活動 |\n|---|---|\n| 7:00-8:00 | 起床・朝食 |"
    )
    diary = await ctx.service.create_diary_with_generation(
        character_id=character.id,
        diary_date=date.today()
    )
    assert diary is not None
    assert diary.character_id == character.id
    assert diary.diary_date == date.today()


# System prompt tests

@pytest.mark.asyncio
async def test_get_system_prompt(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 |\n|---|---|\n| 7:00-8:00 | 起床 |"
    )
    system_prompt = await ctx.service.get_system_prompt(
        character_id=character.id,
        system_prompt_params={"username": "テストユーザー"}
    )
    assert "テストユーザー" in system_prompt
    assert "吹奏楽部" in system_prompt


@pytest.mark.asyncio
async def test_get_system_prompt_cache(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 |\n|---|---|\n| 7:00-8:00 | 起床 |"
    )

    # First call - should build and cache
    prompt1 = await ctx.service.get_system_prompt(
        character_id=character.id,
        system_prompt_params={"username": "User1"}
    )

    # Second call - should use cache
    prompt2 = await ctx.service.get_system_prompt(
        character_id=character.id,
        system_prompt_params={"username": "User2"}
    )

    # Usernames should differ, but base content should be same
    assert "User1" in prompt1
    assert "User2" in prompt2

    # Cache should exist
    assert character.id in ctx.service._system_prompt_cache


@pytest.mark.asyncio
async def test_get_system_prompt_refresh_cache(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 |\n|---|---|\n| 7:00-8:00 | 起床 |"
    )

    # First call
    await ctx.service.get_system_prompt(
        character_id=character.id,
        system_prompt_params={"username": "User1"}
    )

    # Call with refresh_cache=True
    prompt = await ctx.service.get_system_prompt(
        character_id=character.id,
        system_prompt_params={"username": "User1"},
        refresh_cache=True
    )

    assert prompt is not None
    assert character.id in ctx.service._system_prompt_cache


# Batch operation tests

@pytest.mark.asyncio
async def test_create_activity_range_with_generation(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    )

    start_date = date.today() - timedelta(days=2)
    end_date = date.today()

    results = await ctx.service.create_activity_range_with_generation(
        character_id=character.id,
        start_date=start_date,
        end_date=end_date
    )

    assert len(results) == 3  # 3 days
    for i, result in enumerate(results):
        expected_date = start_date + timedelta(days=i)
        assert isinstance(result, ActivityRangeResult)
        assert result.target_date == expected_date
        assert result.daily_schedule is not None
        assert result.diary is not None
        assert result.is_schedule_generated is True
        assert result.is_diary_generated is True


@pytest.mark.asyncio
async def test_create_activity_range_with_generation_skip_existing(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    )

    # Pre-create schedule and diary for today
    today = date.today()
    await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=today,
        content="Existing schedule"
    )
    await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=today,
        content="Existing diary",
        content_context={}
    )

    start_date = date.today() - timedelta(days=1)
    end_date = date.today()

    results = await ctx.service.create_activity_range_with_generation(
        character_id=character.id,
        start_date=start_date,
        end_date=end_date,
        overwrite=False
    )

    assert len(results) == 2

    # Yesterday should be generated
    assert results[0].target_date == start_date
    assert results[0].is_schedule_generated is True
    assert results[0].is_diary_generated is True

    # Today should use existing
    assert results[1].target_date == today
    assert results[1].is_schedule_generated is False
    assert results[1].is_diary_generated is False
    assert results[1].daily_schedule.content == "Existing schedule"
    assert results[1].diary.content == "Existing diary"


@pytest.mark.asyncio
async def test_create_activity_range_with_generation_overwrite(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    )

    # Pre-create schedule and diary for today
    today = date.today()
    await ctx.service.activity.create_daily_schedule(
        character_id=character.id,
        schedule_date=today,
        content="Old schedule"
    )
    await ctx.service.activity.create_diary(
        character_id=character.id,
        diary_date=today,
        content="Old diary",
        content_context={}
    )

    results = await ctx.service.create_activity_range_with_generation(
        character_id=character.id,
        start_date=today,
        end_date=today,
        overwrite=True
    )

    assert len(results) == 1
    assert results[0].target_date == today
    assert results[0].is_schedule_generated is True
    assert results[0].is_diary_generated is True
    # Content should be different from old values (generated new)
    assert results[0].daily_schedule.content != "Old schedule"
    assert results[0].diary.content != "Old diary"


@pytest.mark.asyncio
async def test_create_activity_range_with_generation_default_end_date(ctx):
    character = await ctx.create_character(
        prompt="高校2年生の女子。部活は吹奏楽部でフルートを担当。"
    )
    await ctx.service.activity.create_weekly_schedule(
        character_id=character.id,
        content="| 時間帯 | 月曜 | 火曜 |\n|---|---|---|\n| 7:00-8:00 | 起床 | 起床 |"
    )

    start_date = date.today()

    # end_date not specified, should default to today
    results = await ctx.service.create_activity_range_with_generation(
        character_id=character.id,
        start_date=start_date
    )

    assert len(results) == 1
    assert results[0].target_date == date.today()
