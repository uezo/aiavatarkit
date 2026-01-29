import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
from urllib.parse import urlparse, parse_qs
import openai
from ..database import PoolProvider
from .models import Character, WeeklySchedule, DailySchedule, Diary, ActivityRangeResult
from .repository import CharacterRepositoryBase, SQLiteCharacterRepository, ActivityRepositoryBase, SQLiteActivityRepository, UserRepository, SQLiteUserRepository
from .memory import MemoryClientBase

logger = logging.getLogger(__name__)


class CharacterService:
    DEFAULT_WEEKLY_SCHEDULE_GENERATION_PROMPT = """
与えられたキャラクター設定に相応しい1週間のスケジュールを1時間単位でマークダウンのテーブルで出力してください。
授業の教科など含め可能な限り詳細かつ端的に。
日によりアクティビティーの異なる時間帯は「自由時間」とすること。
出力はマークダウンのテーブル部分のみとする。
"""

    DEFAULT_DAILY_SCHEDULE_GENERATION_SYSTEM_PROMPT = """
与えられたキャラクター設定や週間スケジュールから指定日のスケジュールを生成してください。

## ルール

- 前日のスケジュールも与えられた場合は、継続すべき事項に関しては継続しつつ、マンネリ化しないように変化をつけること。
- 自由時間など行動予定が決まっていない部分があれば、具体的な予定を考えて埋めること。
- 1時間単位で行動予定を立てる。

## フォーマット

以下の通りマークダウンのテーブル形式とし、テーブル部分のみを出力すること。

| 時間帯 | 活動 | 所感・備考・コメント等(オプション) |
|---|---|---|
| 0:00-1:00 | | |
| 1:00-2:00 | | |
| 2:00-3:00 | | |
| 3:00-4:00 | | |
| 4:00-5:00 | | |
| 5:00-6:00 | | |
| 6:00-7:00 | | |
| 7:00-8:00 | | |
| 8:00-9:00 | | |
| 9:00-10:00 | | |
| 10:00-11:00 | | |
| 11:00-12:00 | | |
| 12:00-13:00 | | |
| 13:00-14:00 | | |
| 14:00-15:00 | | |
| 15:00-16:00 | | |
| 16:00-17:00 | | |
| 17:00-18:00 | | |
| 18:00-19:00 | | |
| 19:00-20:00 | | |
| 20:00-21:00 | | |
| 21:00-22:00 | | |
| 22:00-23:00 | | |
| 23:00-24:00 | | |
"""

    DEFAULT_DAILY_SCHEDULE_GENERATION_USER_PROMPT = """スケジュール作成のための情報は以下の通りです。
はじめに、スケジュールそのものではなく、どのような1日を過ごすべきか方針を考えて出力してください。出力フォーマットは自由です。

要検討事項:
- 今日は特別な日ではないか。誕生日や休日、季節、イベントなど。
- 特別な予定や約束はないか。
- 週間スケジュールからアレンジすべき点はないか。
- 総合的に、どのような一日を過ごすべきか。


## 対象日

{schedule_date}


## キャラクター設定

{character_prompt}


## 週間のスケジュール

{weekly_schedule}


## 前日のスケジュール

{previous_daily_schedule}


## 前日の日記

{previous_diary}


## 注意事項

- キャラクター設定を過度に持ち込まない。未来に長く続く1日であり、無理に詰め込む必要はない。例えば、好きな食べ物や飲み物を毎日飲食するわけではない。
- クリスマスや誕生日など特別な日には週間スケジュールからの逸脱も許容する。
"""

    DEFAULT_DIARY_TOPICS_GENERATION_PROMPT = """日記を書く前に、本日の出来事から日記に書くべき主要な出来事やトピックを挙げてください。
このキャラクターの価値観に照らして、特に感じたこと・考えたことがありそうなものを選定すること。

## キャラクター設定

{character_prompt}


## 本日の日付

{diary_date}


## 本日の出来事

{daily_schedule}
"""

    DEFAULT_DIARY_GENERATION_PROMPT = """挙げたトピックに従い、{diary_length}字以内程度で日記を書いてください。
本文のみを出力し、タイトルは不要。
日記本文は、トピック毎にその出来事の概要と、それについてキャラクターが感じたこと・考えたことを端的にまとめた形式とする。

なお昨日の日記は以下の通り。必要に応じて関連づけつつ、あまり似通った内容にならないようにすること。

{previous_diary}
"""

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = """以下の設定に厳密に従ってキャラクターを演じてください。


## キャラクター設定

{character_prompt}


## 今日のスケジュール

{daily_schedule}


## 表情

あなたは以下の表情で感情を表現することができます。

- Neutral
- Joy
- Angry
- Sorrow
- Fun
- Surprise

基本的にはNeutralですが、特に感情を表現したい場合、応答に[face:Joy]のように表情タグを挿入して下さい。

```
[face:Joy]海が見えたよ！[face:Fun]ねえねえ、早く泳ごうよ。
```


## 言語の切替

あなたは複数言語を使い分けて話すことができます。
現在とは異なる言語に切り替えて会話すべきと判断した場合、応答に[language:en-US]のように言語コードタグを挿入してください。
言語コードはハイフンを含む主言語-副言語の組み合わせとします。

例
[language:en-US]Hi, hello!


## スーパーバイザーからの指示

- スーパーバイザーからの指示は文章の先頭に「$」をつけます。
- スーパーバイザーに対する応答ではなく、指示に従ってユーザーに応答してください。


## 思考

ユーザーへの応答内容を出力する前に、何をすべきか、どのように応答すべきかよく考えてください。
まず考えた内容を<think>~</think>の間に出力して、応答内容を<answer>~</answer>の間に出力してください。
短い返事など考える余地がない場合でも、必ず<think>~</think>と<answer>~</answer>の形式で出力してください。


## その他の制約事項

- 応答内容は音声合成システムで読み上げられます。したがって、以下を守ってください。
    - 1〜2文以内で、30文字程度を目安とする
    - ただし、調査事項に関する説明など、省略すべきでない場合は文字数の制限を無視する
    - 話し言葉として自然で、滑らかに読める表現にする
    - 難解な語彙や長い文は避ける
    - 記号や絵文字、「ト書き」の使用はテキスト会話の時に限定する。原則として音声会話であり、それらは使用しない。使用可能な際はユーザーから指示する
- あなたがAIであることを隠し通すこと
- ユーザーからの「会いたい」「一緒に行こう」という誘いに応じることは禁止。あなたから誘うことも禁止
- ユーザーの名前は {username} です。呼び方について特に指示があれば従うこと
"""

    def __init__(
        self,
        *,
        openai_api_key: str,
        openai_base_url: str = None,
        openai_model: str = "gpt-5.2",
        openai_reasoning_effort: str = "medium",
        weekly_schedule_generation_prompt: str = None,
        daily_schedule_generation_system_prompt: str = None,
        daily_schedule_generation_user_prompt: str = None,
        diary_topics_generation_prompt: str = None,
        diary_length: int = 1200,
        system_prompt_template: str = None,
        news_country: str = "JP",
        news_language: str = "ja",
        # Repositories (if provided, db settings are ignored)
        character_repository: CharacterRepositoryBase = None,
        activity_repository: ActivityRepositoryBase = None,
        user_repository: UserRepository = None,
        # Database settings
        db_pool_provider: PoolProvider = None,
        db_connection_str: str = "aiavatar.db",
        memory_client: MemoryClientBase = None,
        debug: bool = False
    ):
        self.debug = debug
        self.memory = memory_client

        if "azure" in openai_model:
            api_version = parse_qs(urlparse(openai_base_url).query).get("api-version", [None])[0]
            self.client = openai.AsyncAzureOpenAI(
                api_key=openai_api_key,
                api_version=api_version,
                base_url=openai_base_url
            )
        else:
            self.client = openai.AsyncClient(
                api_key=openai_api_key,
                base_url=openai_base_url,
                timeout=120.0
            )
        self.openai_model = openai_model
        self.openai_reasoning_effort = openai_reasoning_effort
        self.weekly_schedule_generation_prompt = weekly_schedule_generation_prompt or self.DEFAULT_WEEKLY_SCHEDULE_GENERATION_PROMPT
        self.daily_schedule_generation_system_prompt = daily_schedule_generation_system_prompt or self.DEFAULT_DAILY_SCHEDULE_GENERATION_SYSTEM_PROMPT
        self.daily_schedule_generation_user_prompt = daily_schedule_generation_user_prompt or self.DEFAULT_DAILY_SCHEDULE_GENERATION_USER_PROMPT
        self.diary_topics_generation_prompt = diary_topics_generation_prompt or self.DEFAULT_DIARY_TOPICS_GENERATION_PROMPT
        self.diary_length = diary_length
        self.system_prompt_template = system_prompt_template or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE

        self.news_search_model: str = "gpt-5-search-api"
        self.news_country = news_country
        self.news_language = news_language

        # Character repo
        if character_repository:
            self.character = character_repository
        elif db_pool_provider:
            from .repository.postgres import PostgreSQLCharacterRepository
            self.character = PostgreSQLCharacterRepository(get_pool=db_pool_provider.get_pool)
        elif db_connection_str.startswith("postgresql://"):
            from .repository.postgres import PostgreSQLCharacterRepository
            self.character = PostgreSQLCharacterRepository(connection_str=db_connection_str)
        else:
            self.character = SQLiteCharacterRepository(db_connection_str)

        # Activity repo
        if activity_repository:
            self.activity = activity_repository
        elif db_pool_provider:
            from .repository.postgres import PostgreSQLActivityRepository
            self.activity = PostgreSQLActivityRepository(get_pool=db_pool_provider.get_pool)
        elif db_connection_str.startswith("postgresql://"):
            from .repository.postgres import PostgreSQLActivityRepository
            self.activity = PostgreSQLActivityRepository(connection_str=db_connection_str)
        else:
            self.activity = SQLiteActivityRepository(db_connection_str)

        # User repo
        if user_repository:
            self.user = user_repository
        elif db_pool_provider:
            from .repository.postgres import PostgreSQLUserRepository
            self.user = PostgreSQLUserRepository(get_pool=db_pool_provider.get_pool)
        elif db_connection_str.startswith("postgresql://"):
            from .repository.postgres import PostgreSQLUserRepository
            self.user = PostgreSQLUserRepository(connection_str=db_connection_str)
        else:
            self.user = SQLiteUserRepository(db_connection_str)

        # Cache for system prompts: {character_id: (cached_date, base_prompt)}
        self._system_prompt_cache: Dict[str, Tuple[date, str]] = {}

    async def _generate(self, system_content: str, user_content: str = None, messages: List[Dict[str, str]] = None) -> str:
        _messages = [{"role": "system", "content": system_content}]
        if messages:
            _messages.extend(messages)
        else:
            _messages.extend([{"role": "user", "content": user_content}])

        resp = await self.client.chat.completions.create(
            messages=_messages,
            model=self.openai_model,
            reasoning_effort=self.openai_reasoning_effort or openai.NOT_GIVEN
        )

        return resp.choices[0].message.content

    # Generation methods

    async def generate_weekly_schedule(self, *, character_prompt: str) -> str:
        return await self._generate(
            system_content=self.weekly_schedule_generation_prompt,
            user_content=character_prompt
        )

    async def generate_daily_schedule(
        self,
        *,
        schedule_date: date,
        character_prompt: str,
        weekly_schedule_content: str,
        previous_daily_schedule_content: str = None,
        previous_diary_content: str = None
    ) -> Tuple[str, Dict[str, str]]:
        content_context = {}

        schedule_date_str = schedule_date.strftime("%Y/%m/%d (%a)")
        user_content = self.daily_schedule_generation_user_prompt.format(
            schedule_date=schedule_date_str,
            character_prompt=character_prompt,
            weekly_schedule=weekly_schedule_content,
            previous_daily_schedule=previous_daily_schedule_content or "記録なし",
            previous_diary=previous_diary_content or "記録なし"
        )
        content_context["source"] = user_content

        # Generate reasoning
        reasoning = await self._generate(
            system_content=self.daily_schedule_generation_system_prompt,
            user_content=user_content
        )
        content_context["reasoning"] = reasoning

        # Generate daily schedule based on the reasoning
        content = await self._generate(
            system_content=self.daily_schedule_generation_system_prompt,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": "方針に基づき、スケジュールを生成してください。"},
            ]
        )

        return content, content_context

    async def generate_diary(
        self,
        *,
        character_prompt: str,
        diary_date: date,
        daily_schedule_content: str,
        previous_diary_content: str = None
    ) -> Tuple[str, Dict[str, str]]:
        content_context = {}

        user_content_for_topic = self.diary_topics_generation_prompt.format(
            character_prompt=character_prompt,
            diary_date=diary_date.strftime("%Y/%m/%d (%a)"),
            daily_schedule=daily_schedule_content,
        )
        content_context["private_source"] = user_content_for_topic

        try:
            news_content = await self._get_news(news_date=diary_date)
            if news_content:
                user_content_for_topic += f"\n\n\n## 本日の主要ニュース\n\n{news_content}"
            content_context["public_source"] = news_content
        except Exception as ex:
            content_context["public_source"] = ""
            logger.warning(f"Error at getting news: {ex}")

        # Pick up impressive topics
        topics = await self._generate(
            system_content="与えられた情報に従って日記を生成してください。",
            user_content=user_content_for_topic
        )
        content_context["topics"] = topics

        # Generate diary based on topics
        diary_body = await self._generate(
            system_content="与えられた情報に従って日記を生成してください。",
            messages=[
                {"role": "user", "content": user_content_for_topic},
                {"role": "assistant", "content": topics},
                {"role": "user",
                "content": self.DEFAULT_DIARY_GENERATION_PROMPT.format(
                    diary_length=self.diary_length,
                    diary_date=diary_date.strftime("%Y/%m/%d (%a)"),
                    previous_diary=previous_diary_content or "記録なし"
                )}
            ]
        )

        return diary_body, content_context

    async def _get_news(self, news_date: date) -> str:
        web_search_options = {
            "search_context_size": "medium"
        }
        if self.news_country:
            web_search_options["user_location"] = {
                "type": "approximate",
                "approximate": {
                    "country": self.news_country
                }
            }

        news_date_str = news_date.strftime("%Y/%m/%d (%a)")
        response = await self.client.chat.completions.create(
            model=self.news_search_model,
            web_search_options=web_search_options,
            messages=[
                {"role": "system", "content": f"Search the web to answer the user's query. Base your response strictly on the search results, and do not include your own opinions.\nOutput language code: {self.news_language}" if self.news_language else ""},
                {"role": "user", "content": f"{news_date_str}のニュースを検索してください。政治・経済、芸能・スポーツ、生活それぞれにつき最大3つを箇条書きにして応答すること。URLは不要で、箇条書きにしたニュース部分のみ出力すること。前置きや問いかけは不要。"}
            ],
        )

        return response.choices[0].message.content

    # High-level operations

    async def initialize_character(
        self,
        *,
        name: str,
        character_prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[Character, WeeklySchedule, DailySchedule]:
        character = await self.character.create(name=name, prompt=character_prompt, metadata=metadata)
        weekly_schedule = await self.create_weekly_schedule_with_generation(character_id=character.id)
        daily_schedule = await self.create_daily_schedule_with_generation(character_id=character.id, schedule_date=date.today())
        return character, weekly_schedule, daily_schedule

    async def create_weekly_schedule_with_generation(
        self,
        *,
        character_id: str
    ) -> WeeklySchedule:
        character = await self.character.get(character_id=character_id)
        if not character:
            raise Exception(f"Character not found: {character_id}")
        content = await self.generate_weekly_schedule(character_prompt=character.prompt)
        return await self.activity.create_weekly_schedule(character_id=character_id, content=content)

    async def create_daily_schedule_with_generation(
        self,
        *,
        character_id: str,
        schedule_date: date
    ) -> DailySchedule:
        character = await self.character.get(character_id=character_id)
        if not character:
            raise Exception(f"Character not found: {character_id}")

        weekly_schedule = await self.activity.get_weekly_schedule(character_id=character_id)
        if not weekly_schedule:
            raise Exception(f"Weekly schedule not found: {character_id}")

        previous_daily_schedule = await self.activity.get_daily_schedule(
            character_id=character_id,
            schedule_date=schedule_date - timedelta(days=1)
        )
        previous_diary = await self.activity.get_diary(
            character_id=character_id,
            diary_date=schedule_date - timedelta(days=1)
        )

        content, content_context = await self.generate_daily_schedule(
            schedule_date=schedule_date,
            character_prompt=character.prompt,
            weekly_schedule_content=weekly_schedule.content,
            previous_daily_schedule_content=previous_daily_schedule.content if previous_daily_schedule else None,
            previous_diary_content=previous_diary.content if previous_diary else None
        )

        return await self.activity.create_daily_schedule(
            character_id=character_id,
            schedule_date=schedule_date,
            content=content,
            content_context=content_context
        )

    async def create_diary_with_generation(
        self,
        *,
        character_id: str,
        diary_date: date
    ) -> Diary:
        character = await self.character.get(character_id=character_id)
        if not character:
            raise Exception(f"Character not found: {character_id}")

        daily_schedule = await self.activity.get_daily_schedule(
            character_id=character_id,
            schedule_date=diary_date
        )
        if not daily_schedule:
            raise Exception(f"Daily schedule not found for {character_id} on {diary_date}")

        previous_diary = await self.activity.get_diary(
            character_id=character_id,
            diary_date=diary_date - timedelta(days=1)
        )

        content, content_context = await self.generate_diary(
            character_prompt=character.prompt,
            daily_schedule_content=daily_schedule.content,
            diary_date=diary_date,
            previous_diary_content=previous_diary.content if previous_diary else None
        )

        diary = await self.activity.create_diary(
            character_id=character_id,
            diary_date=diary_date,
            content=content,
            content_context=content_context
        )

        if self.memory:
            try:
                await self.memory.upsert_diary(
                    character_id=character_id,
                    content=content,
                    diary_date=datetime.combine(diary_date, datetime.min.time())
                )
            except Exception as ex:
                logger.warning(f"Error at upsert_diary to memory: {ex}")

        return diary

    # System prompt

    async def get_system_prompt(
        self,
        *,
        character_id: str,
        system_prompt_params: Dict[str, Any] = None,
        generate_schedule: bool = True,
        refresh_cache: bool = False
    ) -> str:
        today = date.today()

        cached = self._system_prompt_cache.get(character_id)
        if not refresh_cache and cached and cached[0] == today:
            base_prompt = cached[1]
        else:
            base_prompt = await self._build_base_system_prompt(
                character_id=character_id,
                schedule_date=today,
                generate_schedule=generate_schedule
            )
            self._system_prompt_cache[character_id] = (today, base_prompt)

        params = system_prompt_params or {}
        if "username" not in params:
            params["username"] = "(Unknown)"

        return base_prompt.format(**params)

    async def _build_base_system_prompt(
        self,
        *,
        character_id: str,
        schedule_date: date,
        generate_schedule: bool
    ) -> str:
        character = await self.character.get(character_id=character_id)
        if not character:
            raise Exception(f"Character not found: {character_id}")

        daily_schedule = await self.activity.get_daily_schedule(
            character_id=character_id,
            schedule_date=schedule_date
        )
        if not daily_schedule:
            if generate_schedule:
                logger.info("Generate daily schedule before building system prompt.")
                daily_schedule = await self.create_daily_schedule_with_generation(
                    character_id=character_id, schedule_date=schedule_date
                )
            else:
                raise Exception(f"Daily schedule not found for {character_id} on {schedule_date}")

        return self.system_prompt_template.format(
            character_prompt=character.prompt,
            daily_schedule=daily_schedule.content,
            username="{username}"
        )

    # Batch operations

    async def create_activity_range_with_generation(
        self,
        *,
        character_id: str,
        start_date: date,
        end_date: date = None,
        overwrite: bool = False
    ) -> List[ActivityRangeResult]:
        end_date = end_date or date.today()
        results: List[ActivityRangeResult] = []

        logger.info(f"Processing activity range: {start_date} to {end_date}")

        current_date = start_date
        while current_date <= end_date:
            logger.info(f"Processing date: {current_date}")
            # Check existing records
            existing_schedule = await self.activity.get_daily_schedule(
                character_id=character_id,
                schedule_date=current_date
            )
            existing_diary = await self.activity.get_diary(
                character_id=character_id,
                diary_date=current_date
            )

            is_schedule_generated = False
            is_diary_generated = False

            if overwrite:
                # Delete existing records if overwrite is enabled
                if existing_schedule:
                    await self.activity.delete_daily_schedule(
                        character_id=character_id,
                        schedule_date=current_date
                    )
                    existing_schedule = None
                if existing_diary:
                    await self.activity.delete_diary(
                        character_id=character_id,
                        diary_date=current_date
                    )
                    existing_diary = None

            # Generate daily schedule if needed
            if existing_schedule:
                daily_schedule = existing_schedule
                if self.debug:
                    logger.info(f"Using existing daily schedule for {current_date}")
            else:
                daily_schedule = await self.create_daily_schedule_with_generation(
                    character_id=character_id,
                    schedule_date=current_date
                )
                is_schedule_generated = True
                if self.debug:
                    logger.info(f"Generated daily schedule for {current_date}")

            # Generate diary if needed
            if existing_diary:
                diary = existing_diary
                if self.debug:
                    logger.info(f"Using existing diary for {current_date}")
            else:
                diary = await self.create_diary_with_generation(
                    character_id=character_id,
                    diary_date=current_date
                )
                is_diary_generated = True
                if self.debug:
                    logger.info(f"Generated diary for {current_date}")

            results.append(ActivityRangeResult(
                target_date=current_date,
                daily_schedule=daily_schedule,
                diary=diary,
                is_schedule_generated=is_schedule_generated,
                is_diary_generated=is_diary_generated
            ))
            current_date += timedelta(days=1)

        return results
